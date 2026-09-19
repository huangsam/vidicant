#!/usr/bin/env python3
"""Generates compile_commands.json for clangd and language servers."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    # Candidate include directories matching build.zig search paths
    candidate_includes = [
        repo_root / "include",
        Path("/opt/homebrew/include"),
        Path("/opt/homebrew/opt/opencv/include/opencv5"),
        Path("/opt/homebrew/opt/opencv/include/opencv4"),
        Path("/opt/homebrew/opt/googletest/include"),
        Path("/opt/homebrew/opt/nlohmann-json/include"),
        Path("/usr/local/include"),
        Path("/usr/local/opt/opencv/include/opencv5"),
        Path("/usr/local/opt/opencv/include/opencv4"),
        Path("/usr/local/opt/googletest/include"),
        Path("/usr/include"),
        Path("/usr/include/opencv4"),
        Path("/usr/include/opencv5"),
    ]

    include_flags = [f"-I{inc.resolve()}" for inc in candidate_includes if inc.is_dir()]
    flags = ["-std=c++17", *include_flags]
    flag_str = " ".join(flags)

    cpp_files: set[Path] = set()
    for pattern in ("src/**/*.cpp", "src/*.cpp", "test/*.cpp"):
        cpp_files.update(repo_root.glob(pattern))

    entries = []
    for file_path in sorted(cpp_files):
        entries.append(
            {
                "directory": str(repo_root),
                "command": f"clang++ {flag_str} -c {file_path.resolve()}",
                "file": str(file_path.resolve()),
            }
        )

    output_path = repo_root / "compile_commands.json"
    output_path.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"Generated {output_path.name} with {len(entries)} translation units.")


if __name__ == "__main__":
    main()
