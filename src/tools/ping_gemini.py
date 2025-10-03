from __future__ import annotations

import argparse
import os
import sys
from typing import Optional


def _load_dotenv_if_present() -> None:
    # simple .env loader (avoids adding python-dotenv)
    env_path = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        # silent; this is a convenience only
        pass


def _get_api_key() -> Optional[str]:
    # support common env var names
    return (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_GENAI_API_KEY")
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Ping Gemini with a test prompt and print the response text")
    parser.add_argument("--prompt", default="Say 'pong' and nothing else.", help="Prompt to send")
    parser.add_argument("--model", default="gemini-1.5-pro-latest", help="Gemini model name")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--max_output_chars", type=int, default=400, help="Trim printed text to this many characters")
    args = parser.parse_args(argv)

    _load_dotenv_if_present()

    key = _get_api_key()
    if not key:
        print("error: missing API key. Set GEMINI_API_KEY or GOOGLE_API_KEY (optionally in .env)", file=sys.stderr)
        print("tip: echo 'GEMINI_API_KEY=your_key_here' > .env", file=sys.stderr)
        return 2

    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        print("error: google-generativeai not installed. Install with: pip install google-generativeai", file=sys.stderr)
        return 3

    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel(args.model)
        resp = model.generate_content(
            args.prompt,
            generation_config={"temperature": args.temperature},
        )
        text = (resp.text or "").strip()
        if not text:
            print("received empty response (no text).", file=sys.stderr)
            # dump minimal diagnostics if present
            try:
                print(getattr(resp, "to_dict", lambda: {} )(), file=sys.stderr)  # type: ignore
            except Exception:
                pass
            return 4
        if len(text) > args.max_output_chars:
            text = text[: args.max_output_chars] + " …"
        print(text)
        return 0
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
