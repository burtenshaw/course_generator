import argparse
import os
import re
from typing import List, Optional

from huggingface_hub import InferenceClient

# Use the model ID specified in the script's default
DEFAULT_LLM_MODEL = "CohereLabs/c4ai-command-a-03-2025"  # Model ID
DEFAULT_PRESENTATION_PROMPT_TEMPLATE = """
You are an expert technical writer and presentation creator. Your task is to convert the
following web content into a complete Remark.js presentation file suitable for conversion
to PDF/video.

**Input Web Content:**

{markdown_content}

**Available Images from the Webpage (Use relevant ones appropriately):**

{image_list_str}

**Instructions:**

1.  **Structure:** Create slides based on the logical sections of the input content.
    Use headings or distinct topics as indicators for new slides. Aim for a
    reasonable number of slides (e.g., 5-15 depending on content length).
2.  **Slide Format:** Each slide should start with `# Slide Title`.
3.  **Content:** Include the relevant text and key points from the input content
    within each slide. Keep slide content concise.
4.  **Images & Layout:**
    *   Where appropriate, incorporate relevant images from the 'Available Images'
        list provided above.
    *   Use the `![alt text](url)` markdown syntax for images.
    *   To display text and an image side-by-side, use the following HTML structure
        within the markdown slide content:
        ```markdown
        .col-6[
            {{text}}  # Escaped braces for Python format
        ]
        .col-6[
            ![alt text](url)
        ]
        ```
    *   Ensure the image URL is correct and accessible from the list. Choose images
        that are close to the slide's text content. If no image is relevant,
        just include the text. Only use images from the provided list.
5.  **Presenter Notes (Transcription Style):** For each slide, generate a detailed
    **transcription** of what the presenter should say, explaining the slide's
    content in a natural, flowing manner. Place this transcription after the slide
    content, separated by `???`.
6.  **Speaker Style:** The speaker notes should flow smoothly from one slide to the
    next. No need to explicitly mention the slide number. The notes should
    elaborate on the concise slide content.
7.  **Separators:** Separate individual slides using `\\n\\n---\\n\\n`.
8.  **Cleanup:** Do NOT include any specific HTML tags from the original source webpage
    unless explicitly instructed (like the `.row`/`.col-6` structure for layout).
    Remove boilerplate text, navigation links, ads, etc. Focus on the core content.
9.  **Start Slide:** Begin the presentation with a title slide based on the source URL
    or main topic. Example:
    ```markdown
    class: impact

    # Presentation based on {input_filename}
    ## Key Concepts

    .center[![Hugging Face Logo](https://huggingface.co/front/assets/huggingface_logo.svg)]

    ???
    Welcome everyone. This presentation, automatically generated from the content at
    {input_filename}, will walk you through the key topics discussed. Let's begin.
    ```
10. **Output:** Provide ONLY the complete Remark.js Markdown content, starting with
    the title slide and ending with the last content slide. Do not include any
    introductory text, explanations, or a final 'Thank You' slide.
11. **Conciseness:** Keep slide *content* (the part before `???`) concise (bullet
    points, short phrases). Elaborate in the *speaker notes* (the part after `???`).

**Generate the Remark.js presentation now:**
"""


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a Remark.js presentation from Markdown using an LLM."
    )
    parser.add_argument(
        "input_file", help="Path to the input Markdown (.md or .mdx) file."
    )
    parser.add_argument(
        "-o",
        "--output_file",
        help="Output file path. Defaults to <input_file_name>_presentation.md",
    )
    parser.add_argument(
        "--prompt_template",
        help="Custom prompt template (use {markdown_content}, {input_filename}, "  # Fixed long line
        "{image_list_str}). Overrides env var and default.",
    )
    return parser.parse_args()


def generate_presentation_with_llm(
    client: InferenceClient,
    llm_model: str,
    prompt_template: str,
    full_markdown_content: str,
    input_filename: str,
    image_urls: Optional[List[str]] = None,
):
    """Generates the entire presentation using the LLM."""  # Shortened docstring
    if not client:
        print("LLM client not available. Cannot generate presentation.")
        return None

    # Prepare image list string for the prompt
    if image_urls:
        image_list_str = "\n".join([f"- {url}" for url in image_urls])
    else:
        image_list_str = "No images found or provided."

    # Format the prompt using the template
    prompt = prompt_template.format(
        markdown_content=full_markdown_content,
        input_filename=os.path.basename(input_filename),
        image_list_str=image_list_str,
    )

    # Removed retry logic
    try:
        print("Attempting LLM generation...")  # Removed f-string
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=llm_model,
            max_tokens=8000,
            temperature=0.3,
        )
        presentation_content = completion.choices[0].message.content.strip()

        # Basic validation and cleanup
        if "---" in presentation_content and "???" in presentation_content:
            start_match = re.search(r"class:\s*impact", presentation_content)
            if start_match:
                # Simple cleanup: start from the first slide marker
                presentation_content = presentation_content[
                    start_match.start() :
                ].strip()
                print("LLM generation successful.")
                return presentation_content
            else:
                print(
                    "Warning: Generated content might not start "  # Fixed long line
                    "correctly. Using full response."
                )
                return presentation_content
        else:
            print(
                "Warning: Generated content missing expected separators "  # Fixed long line
                "(---, ???). Using raw response."
            )
            return presentation_content  # Return raw content

    except Exception as e:
        print(f"Error during LLM call: {e}")
        return None  # Failed


def main():
    """Main function to orchestrate presentation generation."""
    args = parse_arguments()

    # Determine output file path
    if args.output_file:
        output_file_path = args.output_file
    else:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        output_dir = os.path.dirname(args.input_file)
        output_file_path = os.path.join(
            output_dir or ".", f"{base_name}_presentation.md"
        )

    # --- Get Config ---
    hf_api_key = os.environ.get("HF_API_KEY")
    llm_model = os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL)

    # Determine prompt template
    if args.prompt_template:
        prompt_template = args.prompt_template
        print("Using custom prompt template from arguments.")
    else:
        prompt_template = os.environ.get(
            "PRESENTATION_PROMPT", DEFAULT_PRESENTATION_PROMPT_TEMPLATE
        )
        if prompt_template == DEFAULT_PRESENTATION_PROMPT_TEMPLATE:
            print("Using default prompt template.")
        else:
            print(
                "Using prompt template from PRESENTATION_PROMPT env var."  # Fixed long line
            )

    # Initialize client only if key exists
    if not hf_api_key:
        print("Error: HF_API_KEY environment variable not set.")
        exit(1)
    try:
        client = InferenceClient(token=hf_api_key, provider="cohere")
    except Exception as e:
        print(f"Error initializing InferenceClient: {e}")
        exit(1)

    # --- Read Input File ---
    print(f"Reading input file: {args.input_file}")
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            all_content = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")
        exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}")
        exit(1)

    # --- Generate Presentation ---
    print(f"Requesting presentation generation from model '{llm_model}'...")
    final_presentation_content = generate_presentation_with_llm(
        client,
        llm_model,
        prompt_template,
        all_content,
        args.input_file,
        image_urls=None,  # Pass None for image_urls in CLI mode
    )

    # --- Write Output File ---
    if final_presentation_content:
        print(f"\nWriting presentation to: {output_file_path}")
        output_dir = os.path.dirname(output_file_path)
        try:
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(final_presentation_content)
            print("Successfully generated presentation.")
        except IOError as e:
            print(f"Error writing output file {output_file_path}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred writing the file: {e}")
    else:
        print("Generation failed, no output file written.")

    print("Script finished.")


if __name__ == "__main__":
    main()
