#!/usr/bin/env python3
"""
Script to extract speaker notes from presentation.md files and convert them to
audio files.

Usage:
    python transcription_to_audio.py path/to/chapter/directory

This will:
1. Parse the presentation.md file in the specified directory
2. Extract speaker notes (text between ??? markers)
3. Generate audio files using FAL AI with optional voice customization
4. Save audio files in {dir}/audio/{n}.wav format
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import fal_client
import requests
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

VOICE_ID = os.getenv("VOICE_ID")


def extract_speaker_notes(markdown_content):
    """Extract speaker notes from markdown content."""
    # Pattern to match content between ??? markers
    pattern = r"\?\?\?(.*?)(?=\n---|\n$|\Z)"

    # Find all matches using regex
    matches = re.findall(pattern, markdown_content, re.DOTALL)

    # Clean up the extracted notes
    notes = [note.strip() for note in matches]

    return notes


def get_cache_key(text, voice, speed, emotion, language):
    """Generate a unique cache key for the given parameters."""
    # Create a string with all parameters
    params_str = f"{text}|{voice}|{speed}|{emotion}|{language}"

    # Generate a hash of the parameters
    return hashlib.md5(params_str.encode()).hexdigest()


def load_cache(cache_file):
    """Load the cache from a file."""
    if not cache_file.exists():
        return {}

    try:
        with open(cache_file, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Error loading cache: {e}")
        return {}


def save_cache(cache_data, cache_file):
    """Save the cache to a file."""
    try:
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)
    except IOError as e:
        logger.warning(f"Error saving cache: {e}")


def text_to_speech(
    text,
    output_file,
    voice=None,
    speed=1.0,
    emotion="happy",
    language="English",
    cache_dir=None,
):
    """Convert text to speech using FAL AI and save as audio file.

    Args:
        text: The text to convert to speech
        output_file: Path to save the output audio file
        voice: The voice ID to use
        speed: Speech speed (0.5-2.0)
        emotion: Emotion to apply (neutral, happy, sad, etc.)
        language: Language for language boost
        cache_dir: Directory to store cache files
    """
    try:
        start_time = time.monotonic()

        # Create the output directory if it doesn't exist
        output_file.parent.mkdir(exist_ok=True)

        # Set up caching
        cache_file = None
        cache_data = {}
        cache_key = get_cache_key(text, voice, speed, emotion, language)

        if cache_dir:
            cache_dir_path = Path(cache_dir)
            cache_dir_path.mkdir(exist_ok=True)
            cache_file = cache_dir_path / "audio_cache.json"
            cache_data = load_cache(cache_file)

            # Check if we have a cached URL for this request
            if cache_key in cache_data:
                audio_url = cache_data[cache_key]
                logger.info(f"Using cached audio URL: {audio_url}")

                # Download the audio from the cached URL
                response = requests.get(audio_url)
                if response.status_code == 200:
                    with open(output_file, "wb") as f:
                        f.write(response.content)

                    logger.info(f"Downloaded cached audio to {output_file}")
                    return True
                else:
                    logger.warning(f"Cached URL failed, status: {response.status_code}")
                    # Continue with generation as the cached URL failed

        # Set up voice settings
        voice_setting = {"speed": speed, "emotion": emotion}

        # Add custom voice ID if provided
        if voice:
            voice_setting["custom_voice_id"] = voice

        def on_queue_update(update):
            if isinstance(update, fal_client.InProgress):
                for log in update.logs:
                    logger.debug(log["message"])

        # Generate speech with FAL AI
        logger.info(f"Generating speech with voice ID: {voice}")

        result = fal_client.subscribe(
            "fal-ai/minimax-tts/text-to-speech/turbo",
            arguments={
                "text": text,
                "voice_setting": voice_setting,
                "language_boost": language,
            },
            with_logs=True,
            on_queue_update=on_queue_update,
        )

        # Download the audio file from the URL
        if "audio" in result and "url" in result["audio"]:
            audio_url = result["audio"]["url"]
            logger.info(f"Downloading audio from {audio_url}")

            # Cache the URL if caching is enabled
            if cache_file:
                cache_data[cache_key] = audio_url
                save_cache(cache_data, cache_file)
                logger.info(f"Cached audio URL for future use")

            response = requests.get(audio_url)
            if response.status_code == 200:
                # Save the audio file
                with open(output_file, "wb") as f:
                    f.write(response.content)
            else:
                logger.error(f"Failed to download audio: {response.status_code}")
                return False
        else:
            logger.error(f"Unexpected response format: {result}")
            return False

        end_time = time.monotonic()
        logger.info(
            f"Generated audio in {end_time - start_time:.2f} seconds, "
            f"saved to {output_file}"
        )

        return True
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        return False


def process_presentation(
    chapter_dir,
    voice=None,
    speed=1.0,
    emotion="happy",
    language="English",
    cache_dir=None,
):
    """Process the presentation.md file in the given directory."""
    # Construct paths
    chapter_path = Path(chapter_dir)
    presentation_file = chapter_path / "presentation.md"
    audio_dir = chapter_path / "audio"

    # Check if presentation file exists
    if not presentation_file.exists():
        logger.error(f"Presentation file not found: {presentation_file}")
        return False

    # Create audio directory if it doesn't exist
    audio_dir.mkdir(exist_ok=True)

    # Read the presentation file
    with open(presentation_file, "r", encoding="utf-8") as file:
        content = file.read()

    # Extract speaker notes
    notes = extract_speaker_notes(content)

    if not notes:
        logger.warning("No speaker notes found in the presentation file.")
        return False

    logger.info(f"Found {len(notes)} slides with speaker notes.")

    # Generate audio files for each note
    for i, note in enumerate(notes, 1):
        if not note.strip():
            logger.warning(f"Skipping empty note for slide {i}")
            continue

        output_file = audio_dir / f"{i}.wav"
        logger.info(f"Generating audio for slide {i}")

        success = text_to_speech(
            note,
            output_file,
            voice,
            speed,
            emotion,
            language,
            cache_dir,
        )
        if success:
            logger.info(f"Saved audio to {output_file}")
        else:
            logger.error(f"Failed to generate audio for slide {i}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract speaker notes from presentation.md and convert to"
        " audio files."
    )
    parser.add_argument(
        "chapter_dir", help="Path to the chapter directory containing presentation.md"
    )
    parser.add_argument(
        "--voice",
        default=VOICE_ID,
        help="Voice ID to use (defaults to VOICE_ID from .env)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Speech speed (0.5-2.0, default: 1.0)"
    )
    parser.add_argument(
        "--emotion",
        default="happy",
        help="Emotion to apply (neutral, happy, sad, etc.)",
    )
    parser.add_argument(
        "--language", default="English", help="Language for language boost"
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache",
        help="Directory to store cache files (default: .cache)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable caching of audio URLs"
    )
    args = parser.parse_args()

    # Determine cache directory
    cache_dir = None if args.no_cache else args.cache_dir

    logger.info(f"Processing presentation in {args.chapter_dir}")
    success = process_presentation(
        args.chapter_dir,
        args.voice,
        args.speed,
        args.emotion,
        args.language,
        cache_dir,
    )

    if success:
        logger.info("Audio generation completed successfully.")
    else:
        logger.error("Audio generation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
