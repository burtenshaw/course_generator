import os
import re
from huggingface_hub import InferenceClient
from requests.exceptions import RequestException
import time
import argparse

# Use the model ID specified in the script's default
DEFAULT_LLM_MODEL = "CohereLabs/c4ai-command-a-03-2025"  # Model ID from the error log


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a Remark.js presentation from a Markdown file using an LLM."
    )
    parser.add_argument(
        "input_file", help="Path to the input Markdown (.md or .mdx) file."
    )
    parser.add_argument(
        "-o",
        "--output_file",
        help="Path to the output presentation file. Defaults to <input_file_name>_presentation.md",
    )
    return parser.parse_args()


def read_input_file(filepath):
    """Reads content from the specified file."""
    if not os.path.exists(filepath):
        print(f"Error: Input file not found at {filepath}")
        return None
    print(f"Reading input file: {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None


def generate_presentation_with_llm(
    client, llm_model, full_markdown_content, input_filename
):
    """Generates the entire presentation using the LLM."""
    if not client:
        print("LLM client not available. Cannot generate presentation.")
        return None

    # Limit input content length if necessary (though models like Command R+ handle large contexts)
    # max_input_len = 100000 # Example limit
    # if len(full_markdown_content) > max_input_len:
    #     print(f"Warning: Input content truncated to {max_input_len} characters for LLM.")
    #     full_markdown_content = full_markdown_content[:max_input_len]

    prompt = f"""
You are an expert technical writer and presentation creator. Your task is to convert the following Markdown course material into a complete Remark.js presentation file.

**Input Markdown Content:**

{full_markdown_content}

**Instructions:**

1.  **Structure:** Create slides based on the logical sections of the input markdown. Use `## ` headings in the input as the primary indicator for new slides.
2.  **Slide Format:** Each slide should start with `# Slide Title` derived from the corresponding `## Heading`.
3.  **Content:** Include the relevant text, code blocks (preserving language identifiers like ```python), and lists from the input markdown within each slide.
4.  **Images:** Convert Markdown images `![alt](url)` into Remark.js format: `.center[![alt](url)]`. Ensure the image URL is correct and accessible.
5.  **Presenter Notes:** For each slide, generate concise speaker notes (2-4 sentences) summarizing the key points, definitions, or context. Place these notes after the slide content, separated by `???`.
6.  **Separators:** Separate individual slides using `\n\n---\n\n`.
7.  **Cleanup:** Do NOT include any HTML/MDX specific tags like `<CourseFloatingBanner>`, `<Tip>`, `<Question>`, `<Youtube>`, or internal links like `[[...]]` in the final output. Remove frontmatter if present.
8.  **Start/End:**
    *   Begin the presentation with a title slide:
        ```markdown
        class: impact

        # Presentation based on {os.path.basename(input_filename)}
        ## Generated Presentation

        .center[![Hugging Face Logo](https://huggingface.co/front/assets/huggingface_logo.svg)]

        ???
        This presentation was automatically generated from the content of {os.path.basename(input_filename)}. It covers the key topics discussed in the material.
        ```
    *   End the presentation with a final "Thank You" slide:
        ```markdown
        class: center, middle

        # Thank You!

        ???
        This concludes the presentation generated from the provided material.
        ```
9.  **Output:** Provide ONLY the complete Remark.js Markdown content, starting with the title slide and ending with the thank you slide, with all generated slides in between. Do not include any introductory text or explanations before or after the presentation markdown.

**Generate the Remark.js presentation now:**
"""
    max_retries = 2
    retry_delay = 10  # seconds, generation can take time
    for attempt in range(max_retries):
        try:
            print(
                f"Attempting LLM generation (Attempt {attempt + 1}/{max_retries})... This may take a while."
            )
            # Use the client's chat completion method appropriate for the provider
            # For Cohere provider, it might be client.chat.completions.create or similar
            # Assuming client.chat_completion works based on previous script structure
            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=llm_model,
                max_tokens=8000,  # Increase max_tokens significantly for full presentation (adjust based on model limits)
                temperature=0.3,  # Lower temperature for more deterministic structure following
            )
            presentation_content = completion.choices[0].message.content.strip()

            # Basic validation: Check if it looks like a remark presentation
            if "---" in presentation_content and "???" in presentation_content:
                # Attempt to remove potential preamble/postamble from the LLM response
                # Find the first 'class: impact' and last 'Thank You!' slide markers
                start_match = re.search(r"class:\s*impact", presentation_content)
                # Find the end of the "Thank You" slide block more reliably
                thank_you_slide_end_index = presentation_content.rfind(
                    "\n\n???\n"
                )  # Look for the notes separator of the last slide

                if start_match and thank_you_slide_end_index != -1:
                    start_index = start_match.start()
                    # Find the end of the notes for the thank you slide
                    # Search for the end of the notes block, which might just be the end of the string
                    end_of_notes_pattern = re.compile(
                        r"\n\n(?!(\?\?\?|---))", re.MULTILINE
                    )  # Look for a double newline not followed by ??? or ---
                    end_match = end_of_notes_pattern.search(
                        presentation_content,
                        thank_you_slide_end_index + len("\n\n???\n"),
                    )

                    if end_match:
                        end_index = end_match.start()  # End before the double newline
                    else:  # If no clear end found after notes, take rest of string
                        end_index = len(presentation_content)

                    presentation_content = presentation_content[
                        start_index:end_index
                    ].strip()
                    print("LLM generation successful.")
                    return presentation_content
                elif start_match:  # Fallback if end markers are weird but start is okay
                    presentation_content = presentation_content[
                        start_match.start() :
                    ].strip()
                    print("LLM generation successful (end marker adjustment needed).")
                    return presentation_content
                else:
                    print(
                        "Warning: Generated content might not start correctly. Using full response."
                    )
                    return presentation_content  # Return raw if markers not found

            else:
                print(
                    "Warning: Generated content doesn't seem to contain expected Remark.js separators (---, ???)."
                )
                return presentation_content  # Return raw content for inspection

        except RequestException as e:
            print(f"API Request Error (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached for API request.")
                return None
        except Exception as e:
            print(f"Error during LLM call (Attempt {attempt + 1}/{max_retries}): {e}")
            # Attempt to safely access response details if they exist
            response_details = ""
            if hasattr(e, "response"):
                try:
                    status = getattr(e.response, "status_code", "N/A")
                    text = getattr(e.response, "text", "N/A")
                    response_details = f" (Status: {status}, Body: {text[:500]}...)"  # Limit body length
                except Exception as inner_e:
                    response_details = (
                        f" (Could not parse error response details: {inner_e})"
                    )
            print(f"LLM Call Error: {e}{response_details}")

            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached for LLM call.")
                return None

    print("Failed to generate presentation after multiple retries.")
    return None


def write_output_file(filepath, content):
    """Writes the presentation content to the output file."""
    if content is None:
        print("No content to write.")
        return
    print(f"\nWriting presentation to: {filepath}")
    try:
        # Ensure directory exists
        output_dir = os.path.dirname(filepath)
        if (
            output_dir
        ):  # Ensure output_dir is not empty (happens if writing to current dir)
            os.makedirs(output_dir, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print("Successfully generated presentation.")
    except Exception as e:
        print(f"Error writing output file {filepath}: {e}")


# --- Main Orchestration ---


def main():
    """Main function to orchestrate presentation generation."""
    args = parse_arguments()

    # Determine output file path
    if args.output_file:
        output_file_path = args.output_file
    else:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        # Place output in the same directory as input by default
        output_dir = os.path.dirname(args.input_file)
        # Handle case where input file has no directory path
        output_file_path = os.path.join(
            output_dir or ".", f"{base_name}_presentation.md"
        )

    # Get config
    hf_api_key = os.environ.get(
        "HF_API_KEY",
    )
    llm_model = os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL)

    client = InferenceClient(token=hf_api_key, provider="cohere")

    # Read Input
    all_content = read_input_file(args.input_file)

    if all_content is None:
        exit(1)  # Exit if file reading failed

    # Generate Presentation using LLM
    print(f"Requesting presentation generation from model '{llm_model}'...")
    final_presentation_content = generate_presentation_with_llm(
        client, llm_model, all_content, args.input_file
    )

    # Write Output
    write_output_file(output_file_path, final_presentation_content)

    print("Script finished.")


if __name__ == "__main__":
    main()
