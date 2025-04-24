import argparse
import glob
import os
import sys
from typing import List, Tuple

import natsort
from moviepy import *
from pdf2image import convert_from_path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a video from PDF slides and audio files."
    )
    parser.add_argument(
        "--pdf", required=True, help="Path to the PDF file containing slides"
    )
    parser.add_argument(
        "--audio-dir", required=True, help="Directory containing audio files"
    )
    parser.add_argument(
        "--audio-pattern",
        default="*.wav",
        help="Pattern to match audio files (default: *.wav)",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=1.5,
        help="Buffer time in seconds after each audio clip (default: 1.5)",
    )
    parser.add_argument(
        "--output",
        default="final_presentation.mp4",
        help="Output video filename (default: final_presentation.mp4)",
    )
    parser.add_argument(
        "--fps", type=int, default=5, help="Frame rate of output video (default: 5)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=72,
        help="DPI for PDF to image conversion (default: 120)",
    )
    return parser.parse_args()


def find_audio_files(audio_dir: str, pattern: str) -> List[str]:
    """Find and sort audio files in the specified directory."""
    search_pattern = os.path.join(audio_dir, pattern)
    audio_files = natsort.natsorted(glob.glob(search_pattern))
    return audio_files


def convert_pdf_to_images(pdf_path: str, dpi: int) -> List:
    """Convert PDF pages to images."""
    print(f"Converting PDF '{pdf_path}' to images...")
    try:
        pdf_images = convert_from_path(pdf_path, dpi=dpi)
        print(f"Successfully converted {len(pdf_images)} pages from PDF.")
        return pdf_images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        sys.exit(1)


def create_video_clips(
    pdf_images: List, audio_files: List[str], buffer_seconds: float, output_fps: int
) -> List:
    """Create video clips from images and audio files."""
    video_clips_list = []

    print("\nCreating individual video clips...")
    for i, (img, aud_file) in enumerate(zip(pdf_images, audio_files)):
        print(
            f"Processing pair {i + 1}/{len(pdf_images)}: "
            f"Page {i + 1} + {os.path.basename(aud_file)}"
        )
        try:
            # Load audio to get duration
            audio_clip = AudioFileClip(aud_file)
            audio_duration = audio_clip.duration

            # Calculate target duration for the image clip
            target_duration = audio_duration + buffer_seconds

            # Create a temporary file for the image
            temp_img_path = f"temp_slide_{i + 1}.png"
            img.save(temp_img_path, "PNG")

            # Create video clip from image with the correct duration
            # In MoviePy v2.0+, we use ImageSequenceClip with a single image
            img_clip = ImageSequenceClip([temp_img_path], durations=[target_duration])

            # Set FPS for the individual clip
            img_clip = img_clip.with_fps(output_fps)

            # Set the audio for the image clip
            video_clip_with_audio = img_clip.with_audio(audio_clip)

            video_clips_list.append(video_clip_with_audio)
            print(
                f"  -> Clip created (Audio: {audio_duration:.2f}s + "
                f"Buffer: {buffer_seconds:.2f}s = "
                f"Total: {target_duration:.2f}s)"
            )

        except Exception as e:
            print(f"  Error processing pair {i + 1}: {e}")
            print("  Skipping this pair.")
            # Close clips if they were opened, to release file handles
            if "audio_clip" in locals() and audio_clip:
                audio_clip.close()
            if "img_clip" in locals() and img_clip:
                img_clip.close()
            if "video_clip_with_audio" in locals() and video_clip_with_audio:
                video_clip_with_audio.close()

    return video_clips_list


def concatenate_clips(
    video_clips_list: List, output_file: str, output_fps: int
) -> None:
    """Concatenate video clips and write to output file."""
    if not video_clips_list:
        print("\nNo video clips were successfully created. Exiting.")
        sys.exit(1)

    print(f"\nConcatenating {len(video_clips_list)} clips...")
    final_clip = None
    try:
        final_clip = concatenate_videoclips(video_clips_list, method="compose")

        print(f"Writing final video file: {output_file}...")
        # Write the final video file
        final_clip.write_videofile(
            output_file,
            fps=output_fps,
            codec="libx264",
            audio_codec="aac",
            threads=16,
            # logger=None,  # Suppress verbose output
        )
        print("Final video file written successfully.")

    except Exception as e:
        print(f"\nError during concatenation or writing video file: {e}")
        print("Ensure you have enough free disk space and RAM.")

    finally:
        # Close clips to release resources
        if final_clip:
            final_clip.close()
        for clip in video_clips_list:
            clip.close()


def cleanup_temp_files(pdf_images: List) -> None:
    """Clean up temporary image files."""
    print("\nCleaning up temporary files...")
    for i in range(len(pdf_images)):
        temp_img_path = f"temp_slide_{i + 1}.png"
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)


def main():
    """Main function to run the script."""
    args = parse_arguments()

    # Validate inputs
    if not os.path.exists(args.pdf):
        print(f"Error: PDF file '{args.pdf}' not found.")
        sys.exit(1)

    if not os.path.exists(args.audio_dir):
        print(f"Error: Audio directory '{args.audio_dir}' not found.")
        sys.exit(1)

    # Find audio files
    audio_files = find_audio_files(args.audio_dir, args.audio_pattern)
    if not audio_files:
        print(
            f"Error: No audio files found matching pattern '{args.audio_pattern}' "
            f"in directory '{args.audio_dir}'."
        )
        sys.exit(1)

    # Convert PDF to images
    pdf_images = convert_pdf_to_images(args.pdf, args.dpi)

    # Check if number of PDF pages matches number of audio files
    if len(pdf_images) != len(audio_files):
        print("Error: Mismatched number of files found.")
        print(f"  PDF pages ({len(pdf_images)})")
        print(f"  Audio files ({len(audio_files)}): {audio_files}")
        print("Please ensure you have one corresponding audio file for each PDF page.")
        sys.exit(1)

    print(f"Found {len(pdf_images)} PDF pages with {len(audio_files)} audio files.")

    # Create video clips
    video_clips = create_video_clips(pdf_images, audio_files, args.buffer, args.fps)

    # Concatenate clips and create final video
    concatenate_clips(video_clips, args.output, args.fps)

    # Clean up
    cleanup_temp_files(pdf_images)

    print("\nScript finished.")


if __name__ == "__main__":
    main()
