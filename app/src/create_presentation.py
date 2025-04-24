import argparse
import os
import re
import time

from huggingface_hub import InferenceClient

# Use the model ID specified in the script's default
DEFAULT_LLM_MODEL = "CohereLabs/c4ai-command-a-03-2025"  # Model ID from the error log
DEFAULT_PRESENTATION_PROMPT_TEMPLATE = """
You are an expert technical writer and presentation creator. Your task is to convert the following Markdown course material into a complete Remark.js presentation file.

**Input Markdown Content:**

{markdown_content}

**Instructions:**

1.  **Structure:** Create slides based on the logical sections of the input markdown. Use `## ` headings in the input as the primary indicator for new slides.
2.  **Slide Format:** Each slide should start with `# Slide Title` derived from the corresponding `## Heading`.
3.  **Content:** Include the relevant text, code blocks (preserving language identifiers like ```python), and lists from the input markdown within each slide.
4.  **Images:** Convert Markdown images `![alt](url)` into Remark.js format: `.center[![alt](url)]`. Ensure the image URL is correct and accessible.
5.  **Presenter Notes (Transcription Style):** For each slide, generate a detailed **transcription** of what the presenter should say with the slide's content. This should be flowing text suitable for reading aloud. Place this transcription after the slide content, separated by `???`.
6.  **Speaker Style:** The speaker should flow smoothly from one slide to the next. No need to explicitly mention the slide number or introduce the content directly.
6.  **Separators:** Separate individual slides using `\n\n---\n\n`.
7.  **Cleanup:** Do NOT include any HTML/MDX specific tags like `<CourseFloatingBanner>`, `<Tip>`, `<Question>`, `<Youtube>`, or internal links like `[[...]]`. Remove frontmatter.
8.  **References:** Do not include references to files like `2.mdx`. Instead, refer to the title of the section.
9.  **Start Slide:** Begin the presentation with a title slide:
    ```markdown
    class: impact

    # Presentation based on {input_filename}
    ## Generated Presentation

    .center[![Hugging Face Logo](https://huggingface.co/front/assets/huggingface_logo.svg)]

    ???
    Welcome everyone. This presentation, automatically generated from the course material titled '{input_filename}', will walk you through the key topics discussed in the document. Let's begin.
    ```
10.  **Output:** Provide ONLY the complete Remark.js Markdown content, starting with the title slide and ending with the last content slide. Do not include any introductory text, explanations, or a final 'Thank You' slide.
11.  **Style:** Keep slide content concise and to the point with no paragraphs. Speaker notes can expand the content of the slide further.

**Generate the Remark.js presentation now:**
"""


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
    parser.add_argument(
        "--prompt_template",
        help="Custom prompt template string (use {markdown_content} and {input_filename}). Overrides PRESENTATION_PROMPT env var and default.",
    )
    return parser.parse_args()


def generate_presentation_with_llm(
    client, llm_model, prompt_template, full_markdown_content, input_filename
):
    """Generates the entire presentation using the LLM based on the provided prompt template."""
    if not client:
        print("LLM client not available. Cannot generate presentation.")
        return None

    # Format the prompt using the template
    prompt = prompt_template.format(
        markdown_content=full_markdown_content,
        input_filename=os.path.basename(input_filename),
    )

    # Removed retry logic
    try:
        print(f"Attempting LLM generation...")
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
                    "Warning: Generated content might not start correctly. Using full response."
                )
                return presentation_content
        else:
            print(
                "Warning: Generated content missing expected separators (---, ???). Using raw response."
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
                "Using prompt template from PRESENTATION_PROMPT environment variable."
            )

    client = InferenceClient(token=hf_api_key, provider="cohere")
    # --- Read Input File ---
    print(f"Reading input file: {args.input_file}")
    with open(args.input_file, "r", encoding="utf-8") as f:
        all_content = f.read()

    # --- Generate Presentation ---
    print(f"Requesting presentation generation from model '{llm_model}'...")
    final_presentation_content = generate_presentation_with_llm(
        client, llm_model, prompt_template, all_content, args.input_file
    )

    # --- Write Output File ---
    if final_presentation_content:
        print(f"\nWriting presentation to: {output_file_path}")
        output_dir = os.path.dirname(output_file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(final_presentation_content)
        print("Successfully generated presentation.")
    else:
        print("Generation failed, no output file written.")

    print("Script finished.")


if __name__ == "__main__":
    main()
