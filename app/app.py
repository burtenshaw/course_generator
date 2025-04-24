import logging
import os
import re
import shelve
import shutil
import subprocess
import tempfile
from pathlib import Path

import gradio as gr
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Import functions from your scripts (assuming they are structured appropriately)
# It's often better to refactor scripts into functions for easier import
try:
    from huggingface_hub import InferenceClient
    from src.create_presentation import (
        DEFAULT_LLM_MODEL,
        DEFAULT_PRESENTATION_PROMPT_TEMPLATE,
        generate_presentation_with_llm,
    )
    from src.create_video import (
        cleanup_temp_files,
        concatenate_clips,
        convert_pdf_to_images,
        create_video_clips,
        find_audio_files,
    )
    from src.transcription_to_audio import VOICE_ID, text_to_speech
except ImportError as e:
    print(f"Error importing script functions: {e}")
    print("Please ensure scripts are in the 'src' directory and structured correctly.")
    exit(1)

load_dotenv()

# --- Configuration & Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

HF_API_KEY = os.getenv("HF_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
PRESENTATION_PROMPT = os.getenv(
    "PRESENTATION_PROMPT", DEFAULT_PRESENTATION_PROMPT_TEMPLATE
)
CACHE_DIR = ".cache"  # For TTS caching
URL_CACHE_DIR = ".url_cache"
URL_CACHE_FILE = os.path.join(URL_CACHE_DIR, "presentations_cache")

# Initialize clients (do this once if possible, or manage carefully in functions)
try:
    if HF_API_KEY:
        hf_client = InferenceClient(token=HF_API_KEY, provider="cohere")
    else:
        logger.warning("HF_API_KEY not found. LLM generation will fail.")
        hf_client = None
except Exception as e:
    logger.error(f"Failed to initialize Hugging Face client: {e}")
    hf_client = None

# --- Helper Functions ---


def fetch_webpage_content(url):
    """Fetches and extracts basic text content from a webpage."""
    logger.info(f"Fetching content from: {url}")
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.text, "html.parser")

        # Basic text extraction (can be improved significantly)
        paragraphs = soup.find_all("p")
        headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        list_items = soup.find_all("li")

        content = (
            "\n".join([h.get_text() for h in headings])
            + "\n\n"
            + "\n".join([p.get_text() for p in paragraphs])
            + "\n\n"
            + "\n".join(["- " + li.get_text() for li in list_items])
        )

        # Simple cleanup
        content = re.sub(r"\s\s+", " ", content).strip()
        logger.info(
            f"Successfully fetched and parsed content (length: {len(content)})."
        )
        return content
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error parsing URL {url}: {e}")
        return None


def parse_presentation_markdown(markdown_content):
    """Splits presentation markdown into slides with content and notes."""
    slides = []
    slide_parts = re.split(r"\n\n---\n\n", markdown_content)
    for i, part in enumerate(slide_parts):
        if "???" in part:
            content, notes = part.split("???", 1)
            slides.append({"id": i, "content": content.strip(), "notes": notes.strip()})
        else:
            # Handle slides without notes (like title slide maybe)
            slides.append(
                {
                    "id": i,
                    "content": part.strip(),
                    "notes": "",  # Add empty notes field
                }
            )
    logger.info(f"Parsed {len(slides)} slides from markdown.")
    return slides


def reconstruct_presentation_markdown(slides_data):
    """Reconstructs the markdown string from slide data."""
    full_md = []
    for slide in slides_data:
        slide_md = slide["content"]
        if slide[
            "notes"
        ]:  # Only add notes separator if notes exist and are not just whitespace
            slide_md += f"\n\n???\n{slide['notes'].strip()}"
        full_md.append(slide_md.strip())  # Ensure each slide part is stripped
    return "\n\n---\n\n".join(full_md)


def generate_pdf_from_markdown(markdown_file_path, output_pdf_path):
    """Generates a PDF from a Markdown file using bs export + decktape."""
    logger.info(f"Attempting PDF gen: {markdown_file_path} -> {output_pdf_path}")
    working_dir = os.path.dirname(markdown_file_path)
    markdown_filename = os.path.basename(markdown_file_path)
    html_output_dir_name = "bs_html_output"
    html_output_dir_abs = os.path.join(working_dir, html_output_dir_name)
    expected_html_filename = os.path.splitext(markdown_filename)[0] + ".html"
    generated_html_path_abs = os.path.join(html_output_dir_abs, expected_html_filename)
    pdf_gen_success = False  # Flag to track success

    # ---- Step 1: Generate HTML using bs export ----
    try:
        Path(html_output_dir_abs).mkdir(parents=True, exist_ok=True)
        export_command = ["bs", "export", markdown_filename, "-o", html_output_dir_name]
        logger.info(f"Running: {' '.join(export_command)} in CWD: {working_dir}")
        export_result = subprocess.run(
            export_command,
            cwd=working_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        logger.info("Backslide (bs export) OK.")
        logger.debug(f"bs export stdout:\n{export_result.stdout}")
        logger.debug(f"bs export stderr:\n{export_result.stderr}")

        if not os.path.exists(generated_html_path_abs):
            logger.error(f"Expected HTML not found: {generated_html_path_abs}")
            try:
                files_in_dir = os.listdir(html_output_dir_abs)
                logger.error(f"Files in {html_output_dir_abs}: {files_in_dir}")
            except FileNotFoundError:
                logger.error(
                    f"HTML output directory {html_output_dir_abs} not found after bs run."
                )
            raise FileNotFoundError(
                f"Generated HTML not found: {generated_html_path_abs}"
            )

    except FileNotFoundError:
        logger.error(
            "`bs` command not found. Install backslide (`npm install -g backslide`)."
        )
        raise gr.Error("HTML generation tool (backslide/bs) not found.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Backslide (bs export) failed (code {e.returncode}).")
        logger.error(f"bs stderr:\n{e.stderr}")
        raise gr.Error(f"Backslide HTML failed: {e.stderr[:500]}...")
    except subprocess.TimeoutExpired:
        logger.error("Backslide (bs export) timed out.")
        raise gr.Error("HTML generation timed out (backslide).")
    except Exception as e:
        logger.error(f"Unexpected error during bs export: {e}", exc_info=True)
        raise gr.Error(f"Unexpected error during HTML generation: {e}")

    # ---- Step 2: Generate PDF from HTML using decktape ----
    try:
        Path(output_pdf_path).parent.mkdir(parents=True, exist_ok=True)
        html_file_url = Path(generated_html_path_abs).as_uri()
        decktape_command = ["decktape", html_file_url, str(output_pdf_path)]
        logger.info(f"Running PDF conversion: {' '.join(decktape_command)}")
        decktape_result = subprocess.run(
            decktape_command,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        logger.info("Decktape command executed successfully.")
        logger.debug(f"decktape stdout:\n{decktape_result.stdout}")
        logger.debug(f"decktape stderr:\n{decktape_result.stderr}")

        if os.path.exists(output_pdf_path):
            logger.info(f"PDF generated successfully: {output_pdf_path}")
            pdf_gen_success = True  # Mark as success
            return output_pdf_path
        else:
            logger.error("Decktape command finished but output PDF not found.")
            return None

    except FileNotFoundError:
        logger.error(
            "`decktape` command not found. Install decktape (`npm install -g decktape`)."
        )
        raise gr.Error("PDF generation tool (decktape) not found.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Decktape command failed (code {e.returncode}).")
        logger.error(f"decktape stderr:\n{e.stderr}")
        raise gr.Error(f"Decktape PDF failed: {e.stderr[:500]}...")
    except subprocess.TimeoutExpired:
        logger.error("Decktape command timed out.")
        raise gr.Error("PDF generation timed out (decktape).")
    except Exception as e:
        logger.error(
            f"Unexpected error during decktape PDF generation: {e}", exc_info=True
        )
        raise gr.Error(f"Unexpected error during PDF generation: {e}")
    finally:
        # --- Cleanup HTML output directory ---
        if os.path.exists(html_output_dir_abs):
            try:
                shutil.rmtree(html_output_dir_abs)
                logger.info(f"Cleaned up HTML temp dir: {html_output_dir_abs}")
            except Exception as cleanup_e:
                logger.warning(
                    f"Could not cleanup HTML dir {html_output_dir_abs}: {cleanup_e}"
                )
        # Log final status
        if pdf_gen_success:
            logger.info(f"PDF generation process completed for {output_pdf_path}.")
        else:
            logger.error(f"PDF generation process failed for {output_pdf_path}.")


# --- Helper Function to Read CSS ---
def load_css(css_path="app/template/style.css"):
    """Loads CSS content from a file."""
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"CSS file not found at {css_path}. No custom styles applied.")
        return ""  # Return empty string instead of None
    except Exception as e:
        logger.error(f"Error reading CSS file {css_path}: {e}")
        return ""  # Return empty string on error


# --- Gradio Workflow Functions ---


def step1_fetch_and_generate_presentation(url, progress=gr.Progress(track_tqdm=True)):
    """Fetches content, generates presentation markdown, prepares editor, and copies template. Uses caching based on URL."""
    if not url:
        raise gr.Error("Please enter a URL.")
    logger.info(f"Step 1: Fetching & Generating for {url}")
    gr.Info(f"Starting Step 1: Fetching content from {url}...")

    # --- Cache Check ---
    try:
        os.makedirs(URL_CACHE_DIR, exist_ok=True)  # Ensure cache dir exists
        with shelve.open(URL_CACHE_FILE) as cache:
            if url in cache:
                logger.info(f"Cache hit for URL: {url}")
                progress(0.5, desc="Loading cached presentation...")
                cached_data = cache[url]
                presentation_md = cached_data.get("presentation_md")
                slides_data = cached_data.get("slides_data")

                if presentation_md and slides_data:
                    temp_dir = tempfile.mkdtemp()
                    md_path = os.path.join(temp_dir, "presentation.md")
                    try:
                        with open(md_path, "w", encoding="utf-8") as f:
                            f.write(presentation_md)
                        logger.info(
                            f"Wrote cached presentation to temp file: {md_path}"
                        )

                        # --- Copy Template Directory for Cached Item ---
                        template_src_dir = "app/template"
                        template_dest_dir = os.path.join(temp_dir, "app/template")
                        if os.path.isdir(template_src_dir):
                            try:
                                shutil.copytree(template_src_dir, template_dest_dir)
                                logger.info(
                                    f"Copied template dir to {template_dest_dir} (cached)"
                                )
                            except Exception as copy_e:
                                logger.error(
                                    f"Failed to copy template dir for cache: {copy_e}"
                                )
                                shutil.rmtree(temp_dir)
                                raise gr.Error(f"Failed to prepare template: {copy_e}")
                        else:
                            logger.error(
                                f"Template source dir '{template_src_dir}' not found."
                            )
                            shutil.rmtree(temp_dir)
                            raise gr.Error(
                                f"Required template '{template_src_dir}' not found."
                            )

                        progress(0.9, desc="Preparing editor from cache...")
                        logger.info(f"Using cached data for {len(slides_data)} slides.")
                        # Return updates for the UI state and controls
                        return (
                            temp_dir,
                            md_path,
                            slides_data,
                            gr.update(visible=True),  # editor_column
                            gr.update(
                                visible=True
                            ),  # btn_generate_pdf (Enable PDF button next)
                            gr.update(
                                interactive=False
                            ),  # btn_fetch_generate (disable)
                        )
                    except Exception as e:
                        logger.error(f"Error writing cached markdown: {e}")
                        if os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir)
                else:
                    logger.warning(f"Cache entry for {url} incomplete. Regenerating.")
            # --- Cache Miss or Failed Cache Load ---
            logger.info(f"Cache miss for URL: {url}. Proceeding with generation.")
            progress(0.1, desc="Fetching webpage content...")
            if not hf_client:
                raise gr.Error("LLM Client not initialized. Check API Key.")

            web_content = fetch_webpage_content(url)
            if not web_content:
                raise gr.Error("Failed to fetch or parse content from the URL.")

            progress(0.3, desc="Generating presentation with LLM...")
            try:
                presentation_md = generate_presentation_with_llm(
                    hf_client, LLM_MODEL, PRESENTATION_PROMPT, web_content, url
                )
            except Exception as e:
                logger.error(f"Error during LLM call: {e}", exc_info=True)
                raise gr.Error(f"Failed to generate presentation from LLM: {e}")

            if not presentation_md:
                logger.error("LLM generation returned None.")
                raise gr.Error("LLM generation failed (received None).")

            # Check for basic structure early, but parsing handles final validation
            if "---" not in presentation_md:
                logger.warning(
                    "LLM output missing slide separators ('---'). Parsing might fail."
                )
            if "???" not in presentation_md:
                logger.warning(
                    "LLM output missing notes separators ('???'). Notes might be empty."
                )

            progress(0.7, desc="Parsing presentation slides...")
            slides_data = parse_presentation_markdown(presentation_md)
            if not slides_data:
                logger.error("Parsing markdown resulted in zero slides.")
                raise gr.Error("Failed to parse generated presentation markdown.")

            # Create a temporary directory for this session
            temp_dir = tempfile.mkdtemp()
            md_path = os.path.join(temp_dir, "presentation.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(presentation_md)
            logger.info(f"Presentation markdown saved to temp file: {md_path}")

            # --- Copy Template Directory for New Item ---
            template_src_dir = "template"
            template_dest_dir = os.path.join(temp_dir, "template")
            if os.path.isdir(template_src_dir):
                try:
                    shutil.copytree(template_src_dir, template_dest_dir)
                    logger.info(f"Copied template directory to {template_dest_dir}")
                except Exception as copy_e:
                    logger.error(f"Failed to copy template directory: {copy_e}")
                    shutil.rmtree(temp_dir)
                    raise gr.Error(f"Failed to prepare template: {copy_e}")
            else:
                logger.error(f"Template source dir '{template_src_dir}' not found.")
                shutil.rmtree(temp_dir)
                raise gr.Error(f"Required template '{template_src_dir}' not found.")

            # --- Store in Cache ---
            try:
                with shelve.open(URL_CACHE_FILE) as cache_write:
                    cache_write[url] = {
                        "presentation_md": presentation_md,
                        "slides_data": slides_data,
                    }
                    logger.info(
                        f"Stored generated presentation in cache for URL: {url}"
                    )
            except Exception as e:
                logger.error(f"Failed to write to cache for URL {url}: {e}")

            progress(0.9, desc="Preparing editor...")
            logger.info(f"Prepared data for {len(slides_data)} slides.")

            # Return updates for the UI state and controls
            return (
                temp_dir,
                md_path,
                slides_data,
                gr.update(visible=True),  # editor_column
                gr.update(visible=True),  # btn_generate_pdf (Enable PDF button next)
                gr.update(interactive=False),  # btn_fetch_generate (disable)
            )

    except Exception as e:
        logger.error(f"Error in step 1 (fetch/generate): {e}", exc_info=True)
        raise gr.Error(f"Error during presentation setup: {e}")


def step2_build_slides(
    state_temp_dir,
    state_md_path,
    state_slides_data,
    *editors,
    progress=gr.Progress(track_tqdm=True),
):
    """Renamed from step2_generate_pdf"""
    if not all([state_temp_dir, state_md_path, state_slides_data]):
        raise gr.Error("Session state missing.")
    logger.info("Step 2: Building Slides (PDF + Images)")
    gr.Info("Starting Step 2: Building slides...")
    num_slides = len(state_slides_data)
    MAX_SLIDES = 20
    all_editors = list(editors)
    if len(all_editors) != MAX_SLIDES * 2:
        raise gr.Error(f"Incorrect editor inputs: {len(all_editors)}")
    edited_contents = all_editors[:MAX_SLIDES][:num_slides]
    edited_notes_list = all_editors[MAX_SLIDES:][:num_slides]
    if len(edited_contents) != num_slides or len(edited_notes_list) != num_slides:
        raise gr.Error("Editor input mismatch.")

    progress(0.1, desc="Saving edited markdown...")
    updated_slides = []
    for i in range(num_slides):
        updated_slides.append(
            {"id": i, "content": edited_contents[i], "notes": edited_notes_list[i]}
        )
    updated_md = reconstruct_presentation_markdown(updated_slides)
    try:
        with open(state_md_path, "w", encoding="utf-8") as f:
            f.write(updated_md)
        logger.info(f"Saved edited markdown: {state_md_path}")
    except IOError as e:
        raise gr.Error(f"Failed to save markdown: {e}")

    progress(0.3, desc="Generating PDF...")
    pdf_output_path = os.path.join(state_temp_dir, "presentation.pdf")
    generated_pdf_path = generate_pdf_from_markdown(state_md_path, pdf_output_path)
    if not generated_pdf_path:
        raise gr.Error("PDF generation failed (check logs).")

    progress(0.7, desc="Converting PDF to images...")
    pdf_images = []
    try:
        pdf_images = convert_pdf_to_images(
            generated_pdf_path, dpi=150
        )  # Use generated path
        if not pdf_images:
            raise gr.Error("PDF to image conversion failed.")
        logger.info(f"Converted PDF to {len(pdf_images)} images.")
        if len(pdf_images) != num_slides:
            gr.Warning(
                f"PDF page count ({len(pdf_images)}) != slide count ({num_slides}). Images might mismatch."
            )
            # Pad or truncate? For now, just return what we have, UI update logic handles MAX_SLIDES
    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}", exc_info=True)
        # Proceed without images? Or raise error? Let's raise.
        raise gr.Error(f"Failed to convert PDF to images: {e}")

    info_msg = f"Built {len(pdf_images)} slide images. Ready for Step 3."
    logger.info(info_msg)
    gr.Info(info_msg)
    progress(1.0, desc="Slide build complete.")
    # Return tuple WITHOUT status textbox update
    return (
        generated_pdf_path,
        pdf_images,  # Return the list of image paths
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value=generated_pdf_path, visible=True),
    )


def step3_generate_audio(*args, progress=gr.Progress(track_tqdm=True)):
    """Generates audio files for the speaker notes using edited content."""
    # Args structure:
    # args[0]: state_temp_dir
    # args[1]: state_md_path
    # args[2]: original_slides_data (list of dicts, used to get count)
    # args[3 : 3 + MAX_SLIDES]: values from all_code_editors
    # args[3 + MAX_SLIDES :]: values from all_notes_textboxes

    state_temp_dir = args[0]
    state_md_path = args[1]
    original_slides_data = args[2]
    editors = args[3:]
    num_slides = len(original_slides_data)
    if num_slides == 0:
        logger.error("Step 3 (Audio) called with zero slides data.")
        raise gr.Error("No slide data available. Please start over.")

    MAX_SLIDES = 20  # Ensure this matches UI definition
    code_editors_start_index = 3
    notes_textboxes_start_index = 3 + MAX_SLIDES

    # Slice the *actual* edited values based on num_slides
    edited_contents = args[
        code_editors_start_index : code_editors_start_index + num_slides
    ]
    edited_notes_list = args[
        notes_textboxes_start_index : notes_textboxes_start_index + num_slides
    ]

    if not state_temp_dir or not state_md_path:
        raise gr.Error("Session state lost (Audio step). Please start over.")

    # Check slicing
    if len(edited_contents) != num_slides or len(edited_notes_list) != num_slides:
        logger.error(
            f"Input slicing error (Audio step): Expected {num_slides}, got {len(edited_contents)} contents, {len(edited_notes_list)} notes."
        )
        raise gr.Error(
            f"Input processing error: Mismatch after slicing ({num_slides} slides)."
        )

    logger.info(f"Processing {num_slides} slides for audio generation.")
    audio_dir = os.path.join(state_temp_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    # --- Update the presentation.md file AGAIN in case notes changed after PDF ---
    # This might be redundant if users don't edit notes between PDF and Audio steps,
    # but ensures the audio matches the *latest* notes displayed.
    progress(0.1, desc="Saving latest notes...")
    updated_slides_data = []
    for i in range(num_slides):
        updated_slides_data.append(
            {
                "id": original_slides_data[i]["id"],  # Keep original ID
                "content": edited_contents[i],  # Use sliced edited content
                "notes": edited_notes_list[i],  # Use sliced edited notes
            }
        )

    updated_markdown = reconstruct_presentation_markdown(updated_slides_data)
    try:
        with open(state_md_path, "w", encoding="utf-8") as f:
            f.write(updated_markdown)
        logger.info(f"Updated presentation markdown before audio gen: {state_md_path}")
    except IOError as e:
        logger.error(f"Failed to save updated markdown before audio gen: {e}")
        # Continue with audio gen, but log warning
        gr.Warning(f"Could not save latest notes to markdown file: {e}")

    generated_audio_paths = ["" for _ in range(num_slides)]
    audio_generation_failed = False
    successful_audio_count = 0

    for i in range(num_slides):
        note_text = edited_notes_list[i]
        slide_num = i + 1
        progress(
            (i + 1) / num_slides * 0.8 + 0.1,
            desc=f"Audio slide {slide_num}/{num_slides}",
        )
        output_file_path = Path(audio_dir) / f"{slide_num}.wav"
        if not note_text or not note_text.strip():
            try:  # Generate silence
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "anullsrc=r=44100:cl=mono",
                        "-t",
                        "0.1",
                        "-q:a",
                        "9",
                        str(output_file_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                generated_audio_paths[i] = str(output_file_path)
            except Exception as e:
                audio_generation_failed = True
                logger.error(f"Silence gen failed slide {i + 1}: {e}")
            continue
        try:  # Generate TTS
            success = text_to_speech(
                note_text, output_file_path, voice=VOICE_ID, cache_dir=CACHE_DIR
            )
            if success:
                generated_audio_paths[i] = str(output_file_path)
                successful_audio_count += 1
            else:
                audio_generation_failed = True
                logger.error(f"TTS failed slide {i + 1}")
        except Exception as e:
            audio_generation_failed = True
            logger.error(f"TTS exception slide {i + 1}: {e}", exc_info=True)

    # --- Prepare outputs for Gradio ---
    audio_player_updates = [
        gr.update(value=p if p else None, visible=bool(p and os.path.exists(p)))
        for p in generated_audio_paths
    ]
    regen_button_updates = [gr.update(visible=True)] * num_slides
    audio_player_updates.extend(
        [gr.update(value=None, visible=False)] * (MAX_SLIDES - num_slides)
    )
    regen_button_updates.extend([gr.update(visible=False)] * (MAX_SLIDES - num_slides))

    info_msg = f"Generated {successful_audio_count}/{num_slides} audio clips. "
    if audio_generation_failed:
        info_msg += "Some audio failed. Review/Regenerate before video."
        gr.Warning(info_msg)
    else:
        info_msg += "Ready for Step 4."
        gr.Info(info_msg)
    logger.info(info_msg)
    progress(1.0, desc="Audio generation complete.")

    # Return tuple WITHOUT status textbox update
    return (
        audio_dir,
        gr.update(visible=True),  # btn_generate_video
        gr.update(visible=False),  # btn_generate_audio
        *audio_player_updates,
        *regen_button_updates,
    )


def step4_generate_video(
    state_temp_dir,
    state_audio_dir,
    state_pdf_path,  # Use PDF path from state
    progress=gr.Progress(track_tqdm=True),
):
    """Generates the final video using PDF images and audio files."""
    if not state_temp_dir or not state_audio_dir or not state_pdf_path:
        raise gr.Error("Session state lost (Video step). Please start over.")
    if not os.path.exists(state_pdf_path):
        raise gr.Error(f"PDF file not found: {state_pdf_path}. Cannot generate video.")
    if not os.path.isdir(state_audio_dir):
        raise gr.Error(
            f"Audio directory not found: {state_audio_dir}. Cannot generate video."
        )

    video_output_path = os.path.join(state_temp_dir, "final_presentation.mp4")

    progress(0.1, desc="Preparing video components...")
    pdf_images = []  # Initialize to ensure cleanup happens
    try:
        # Find audio files (natsorted)
        audio_files = find_audio_files(state_audio_dir, "*.wav")
        if not audio_files:
            logger.warning(
                f"No WAV files found in {state_audio_dir}. Video might lack audio."
            )
            # Decide whether to proceed with silent video or error out
            # raise gr.Error(f"No audio files found in {state_audio_dir}")

        # Convert PDF to images
        progress(0.2, desc="Converting PDF to images...")
        pdf_images = convert_pdf_to_images(state_pdf_path, dpi=150)
        if not pdf_images:
            raise gr.Error(f"Failed to convert PDF ({state_pdf_path}) to images.")

        # Allow video generation even if audio is missing or count mismatch
        # The create_video_clips function should handle missing audio gracefully (e.g., use image duration)
        if len(pdf_images) != len(audio_files):
            logger.warning(
                f"Mismatch: {len(pdf_images)} PDF pages vs {len(audio_files)} audio files. Video clips might have incorrect durations or missing audio."
            )
            # Pad the shorter list? For now, let create_video_clips handle it.

        progress(0.5, desc="Creating individual video clips...")
        buffer_seconds = 1.0
        output_fps = 10
        video_clips = create_video_clips(
            pdf_images, audio_files, buffer_seconds, output_fps
        )

        if not video_clips:
            raise gr.Error("Failed to create any video clips.")

        progress(0.8, desc="Concatenating clips...")
        concatenate_clips(video_clips, video_output_path, output_fps)

        logger.info(f"Video concatenation complete: {video_output_path}")

        progress(0.95, desc="Cleaning up temp images...")
        cleanup_temp_files(pdf_images)  # Pass the list of image paths

    except Exception as e:
        if pdf_images:
            cleanup_temp_files(pdf_images)
        logger.error(f"Video generation failed: {e}", exc_info=True)
        raise gr.Error(f"Video generation failed: {e}")

    info_msg = f"Video generated: {os.path.basename(video_output_path)}"
    logger.info(info_msg)
    gr.Info(info_msg)
    progress(1.0, desc="Video Complete.")
    # Return tuple WITHOUT status textbox update
    return (
        gr.update(value=video_output_path, visible=True),  # video_output
        gr.update(visible=False),  # btn_generate_video
    )


def cleanup_session(temp_dir):
    """Removes the temporary directory."""
    if temp_dir and isinstance(temp_dir, str) and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
            return "Cleaned up session files."
        except Exception as e:
            logger.error(f"Error cleaning up temp directory {temp_dir}: {e}")
            return f"Error during cleanup: {e}"
    logger.warning(f"Cleanup called but temp_dir invalid or not found: {temp_dir}")
    return "No valid temporary directory found to clean."


# --- Gradio Interface ---

# Load custom CSS
custom_css = load_css()

with gr.Blocks(
    theme=gr.themes.Soft(), css=custom_css, title="Webpage to Video"
) as demo:
    gr.Markdown("# Webpage to Video Presentation Generator")

    # State variables
    state_temp_dir = gr.State(None)
    state_md_path = gr.State(None)
    state_audio_dir = gr.State(None)
    state_pdf_path = gr.State(None)
    state_slides_data = gr.State([])
    state_pdf_image_paths = gr.State([])

    MAX_SLIDES = 20

    # --- Tabbed Interface ---
    with gr.Tabs(elem_id="tabs") as tabs_widget:
        # Tab 1: Generate Presentation
        with gr.TabItem("1. Generate Presentation", id=0):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Step 1:** Enter URL")
                    input_url = gr.Textbox(
                        label="Webpage URL",
                        value="https://huggingface.co/blog/llm-course",
                    )
                    btn_fetch_generate = gr.Button(
                        value="1. Fetch & Generate", variant="primary"
                    )
                with gr.Column(scale=4):
                    gr.Markdown(
                        "### Instructions\n1. Enter URL & click 'Fetch & Generate'.\n2. Editor appears below tabs.\n3. Go to next tab."
                    )

        # Tab 2: Build Slides
        with gr.TabItem("2. Build Slides", id=1):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Step 2:** Review/Edit, then build slides.")
                    btn_build_slides = gr.Button(
                        value="2. Build Slides", variant="secondary", visible=False
                    )
                    pdf_download_link = gr.File(
                        label="Download PDF", visible=False, interactive=False
                    )
                with gr.Column(scale=4):
                    gr.Markdown(
                        "### Instructions\n1. Edit content/notes below.\n2. Click 'Build Slides'. Images appear.\n3. Download PDF from sidebar.\n4. Go to next tab."
                    )

        # Tab 3: Generate Audio
        with gr.TabItem("3. Generate Audio", id=2):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Step 3:** Review/Edit notes, then generate audio.")
                    btn_generate_audio = gr.Button(
                        value="3. Generate Audio", variant="primary", visible=False
                    )
                with gr.Column(scale=4):
                    gr.Markdown(
                        "### Instructions\n1. Finalize notes below.\n2. Click 'Generate Audio'.\n3. Regenerate if needed.\n4. Go to next tab."
                    )

        # Tab 4: Generate Video
        with gr.TabItem("4. Create Video", id=3):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Step 4:** Create the final video.")
                    btn_generate_video = gr.Button(
                        value="4. Create Video", variant="primary", visible=False
                    )
                with gr.Column(scale=4):
                    gr.Markdown(
                        "### Instructions\n1. Click 'Create Video'.\n2. Video appears below."
                    )
                    video_output = gr.Video(label="Final Video", visible=False)

    # Define the shared editor structure once, AFTER tabs
    slide_editors_group = []
    with gr.Column(visible=False) as editor_column:  # Initially hidden
        gr.Markdown("--- \n## Edit Slides & Notes")
        gr.Markdown("_(PDF uses content & notes, Audio uses notes only)_")
        for i in range(MAX_SLIDES):
            with gr.Accordion(f"Slide {i + 1}", open=(i == 0), visible=False) as acc:
                with gr.Row():  # Row for Content/Preview/Image
                    with gr.Column(scale=1):
                        code_editor = gr.Code(
                            label="Content (Markdown)",
                            language="markdown",
                            lines=15,
                            interactive=True,
                            visible=False,
                        )
                        notes_textbox = gr.Code(
                            label="Script/Notes (for Audio)",
                            lines=8,
                            language="markdown",
                            interactive=True,
                            visible=False,
                        )
                    with gr.Column(scale=1):
                        slide_image = gr.Image(
                            label="Slide Image",
                            visible=False,
                            interactive=False,
                            height=300,
                        )
                        md_preview = gr.Markdown(visible=False)
                with gr.Row():  # Row for audio controls
                    audio_player = gr.Audio(
                        label="Generated Audio",
                        visible=False,
                        interactive=False,
                        scale=3,
                    )
                    regen_button = gr.Button(
                        value="Regen Audio", visible=False, scale=1, size="sm"
                    )
                slide_editors_group.append(
                    (
                        acc,
                        code_editor,
                        md_preview,
                        notes_textbox,
                        audio_player,
                        regen_button,
                        slide_image,
                    )
                )
                code_editor.change(
                    fn=lambda x: x,
                    inputs=code_editor,
                    outputs=md_preview,
                    show_progress="hidden",
                )

    # --- Component Lists for Updates ---
    all_editor_components = [comp for group in slide_editors_group for comp in group]
    all_code_editors = [group[1] for group in slide_editors_group]
    all_notes_textboxes = [group[3] for group in slide_editors_group]
    all_audio_players = [group[4] for group in slide_editors_group]
    all_regen_buttons = [group[5] for group in slide_editors_group]
    all_slide_images = [group[6] for group in slide_editors_group]

    # --- Function to regenerate audio --- (Assumed correct)
    # ... (regenerate_single_audio implementation as fixed before)...
    def regenerate_single_audio(
        slide_idx, note_text, temp_dir, progress=gr.Progress(track_tqdm=True)
    ):
        # ...(Implementation as fixed before)...
        if (
            not temp_dir
            or not isinstance(temp_dir, str)
            or not os.path.exists(temp_dir)
        ):
            logger.error(f"Regen audio failed: Invalid temp_dir '{temp_dir}'")
            return gr.update(value=None, visible=False)
        slide_num = slide_idx + 1
        audio_dir = os.path.join(temp_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        output_file = Path(audio_dir) / f"{slide_num}.wav"
        logger.info(f"Regenerating audio for slide {slide_num} -> {output_file}")
        progress(0.1, desc=f"Regen audio slide {slide_num}...")
        if not note_text or not note_text.strip():
            logger.warning(f"Note for slide {slide_num} empty. Generating silence.")
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "anullsrc=r=44100:cl=mono",
                        "-t",
                        "0.1",
                        "-q:a",
                        "9",
                        str(output_file),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.info(f"Created silent placeholder: {output_file}")
                progress(1.0, desc=f"Generated silence slide {slide_num}.")
                return gr.update(value=str(output_file), visible=True)
            except Exception as e:
                logger.error(f"Failed silent gen slide {slide_num}: {e}")
                return gr.update(value=None, visible=False)
        else:
            try:
                success = text_to_speech(
                    note_text, output_file, voice=VOICE_ID, cache_dir=CACHE_DIR
                )
                if success:
                    logger.info(f"Regen OK slide {slide_num}")
                    progress(1.0, desc=f"Audio regen OK slide {slide_num}.")
                    return gr.update(value=str(output_file), visible=True)
                else:
                    logger.error(f"Regen TTS failed slide {slide_num}")
                    return gr.update(value=None, visible=False)
            except Exception as e:
                logger.error(
                    f"Regen TTS exception slide {slide_num}: {e}", exc_info=True
                )
                return gr.update(value=None, visible=False)

    # --- Connect the individual Re-generate buttons ---
    # Update unpacking to include slide_image (7 items)
    for i, (
        acc,
        code_edit,
        md_preview,
        notes_tb,
        audio_pl,
        regen_btn,
        slide_image,
    ) in enumerate(slide_editors_group):
        regen_btn.click(
            fn=regenerate_single_audio,
            inputs=[gr.State(i), notes_tb, state_temp_dir],
            outputs=[audio_pl],
            show_progress="minimal",
        )

    # --- Main Button Click Handlers --- (Outputs use locally defined component vars)

    # Step 1 Click Handler
    step1_outputs = [
        state_temp_dir,
        state_md_path,
        state_slides_data,
        editor_column,  # Show the editor column
        btn_build_slides,  # Enable the button in Tab 2
        btn_fetch_generate,  # Disable self
    ]
    btn_fetch_generate.click(
        fn=step1_fetch_and_generate_presentation,
        inputs=[input_url],
        outputs=step1_outputs,
        show_progress="full",
    ).then(
        fn=lambda s_data: [
            upd
            for i, slide in enumerate(s_data)
            if i < MAX_SLIDES
            for upd in [
                gr.update(
                    label=f"Slide {i + 1}: {slide['content'][:25]}...",
                    visible=True,
                    open=(i == 0),
                ),  # Accordion
                gr.update(value=slide["content"], visible=True),  # Code Editor
                gr.update(value=slide["content"], visible=True),  # MD Preview
                gr.update(value=slide["notes"], visible=True),  # Notes Textbox
                gr.update(value=None, visible=False),  # Audio Player
                gr.update(visible=False),  # Regen Button
                gr.update(value=None, visible=False),  # Slide Image
            ]
        ]
        + [
            upd
            for i in range(len(s_data), MAX_SLIDES)
            for upd in [gr.update(visible=False)] * 7
        ],
        inputs=[state_slides_data],
        outputs=all_editor_components,
        show_progress="hidden",
    ).then(lambda: gr.update(selected=1), outputs=tabs_widget)  # Switch to Tab 2

    # Step 2 Click Handler
    step2_inputs = (
        [state_temp_dir, state_md_path, state_slides_data]
        + all_code_editors
        + all_notes_textboxes
    )
    step2_outputs = [
        state_pdf_path,
        state_pdf_image_paths,
        btn_generate_audio,  # Enable button in Tab 3
        btn_build_slides,  # Disable self
        pdf_download_link,  # Update download link in Tab 2
    ]
    btn_build_slides.click(
        fn=step2_build_slides,
        inputs=step2_inputs,
        outputs=step2_outputs,
        show_progress="full",
    ).then(
        fn=lambda image_paths: [
            gr.update(
                value=image_paths[i] if i < len(image_paths) else None,
                visible=(i < len(image_paths)),
            )
            for i in range(MAX_SLIDES)
        ],
        inputs=[state_pdf_image_paths],
        outputs=all_slide_images,
        show_progress="hidden",
    ).then(lambda: gr.update(selected=2), outputs=tabs_widget)  # Switch to Tab 3

    # Step 3 Click Handler
    step3_inputs = (
        [state_temp_dir, state_md_path, state_slides_data]
        + all_code_editors
        + all_notes_textboxes
    )
    step3_outputs = (
        [
            state_audio_dir,
            btn_generate_video,  # Enable button in Tab 4
            btn_generate_audio,  # Disable self
        ]
        + all_audio_players
        + all_regen_buttons
    )
    btn_generate_audio.click(
        fn=step3_generate_audio,
        inputs=step3_inputs,
        outputs=step3_outputs,
        show_progress="full",
    ).then(lambda: gr.update(selected=3), outputs=tabs_widget)  # Switch to Tab 4

    # Step 4 Click Handler
    step4_inputs = [state_temp_dir, state_audio_dir, state_pdf_path]
    step4_outputs = [
        video_output,  # Update video output in Tab 4
        btn_generate_video,  # Disable self
    ]
    btn_generate_video.click(
        fn=step4_generate_video,
        inputs=step4_inputs,
        outputs=step4_outputs,
        show_progress="full",
    )

if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(URL_CACHE_DIR, exist_ok=True)
    demo.queue().launch(debug=True)
