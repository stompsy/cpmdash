import os
import re
from pathlib import Path
from typing import Any

from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.templatetags.static import static


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1048576:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / 1048576:.1f} MB"


def _process_html_content(content: str) -> str:
    """Process HTML content to fix image sources for Django static files."""
    # Pattern to match img src attributes
    img_pattern = r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>'

    def replace_src(match: re.Match[str]) -> str:
        img_tag = match.group(0)
        src_value = match.group(1)

        # If it's already a Django static URL or absolute URL, leave it alone
        if src_value.startswith("/static/") or src_value.startswith("http"):
            return img_tag

        # If it's a relative path, assume it's in the media folder
        if not src_value.startswith("/"):
            # Convert relative paths to media folder paths
            # Handle different image path patterns
            if src_value.startswith("./images/"):
                # Remove ./images/ prefix to get the relative path within media
                media_path = src_value.replace("./images/", "", 1)
            elif src_value.startswith("./"):
                # Remove ./ prefix for other relative paths
                media_path = src_value[2:]
            else:
                # For other relative paths, use as-is
                media_path = src_value

            static_url = static(f"media/{media_path}")
            return img_tag.replace(f'src="{src_value}"', f'src="{static_url}"').replace(
                f"src='{src_value}'", f"src='{static_url}'"
            )

        return img_tag

    # Process all img tags
    processed_content = re.sub(img_pattern, replace_src, content)
    return processed_content


def list_partials(request: HttpRequest) -> HttpResponse:
    """List all template partials grouped by category."""
    partials_dir = Path(settings.BASE_DIR) / "src" / "templates" / "partials"
    categories: dict[str, list[dict[str, Any]]] = {}

    for root, _dirs, files in os.walk(partials_dir):
        for file in files:
            if file.endswith(".html"):
                rel_path = Path(root).relative_to(partials_dir)
                category = "Root" if rel_path == Path(".") else str(rel_path)

                if category not in categories:
                    categories[category] = []

                categories[category].append(
                    {
                        "name": file,
                        "path": str(rel_path / file) if rel_path != Path(".") else file,
                        "full_path": os.path.join(root, file),
                        "size": os.path.getsize(os.path.join(root, file)),
                        "size_formatted": _format_file_size(
                            os.path.getsize(os.path.join(root, file))
                        ),
                    }
                )

    # Sort categories and files within each category
    sorted_categories: dict[str, list[dict[str, Any]]] = {}
    for category in sorted(categories.keys()):
        sorted_categories[category] = sorted(categories[category], key=lambda x: x["name"])

    return render(
        request,
        "partials_viewer/list.html",
        {
            "categories": sorted_categories,
            "total_partials": sum(len(partials) for partials in sorted_categories.values()),
        },
    )


def view_partial(request: HttpRequest, partial_path: str) -> HttpResponse:
    """View a specific partial."""
    partials_dir = Path(settings.BASE_DIR) / "src" / "templates" / "partials"
    full_path = partials_dir / partial_path

    if not full_path.exists() or not str(full_path).startswith(str(partials_dir)):
        return render(request, "partials_viewer/error.html", {"error": "Partial not found"})

    with open(full_path) as f:
        content = f.read()

    # Process the content to fix image sources
    processed_content = _process_html_content(content)

    return render(
        request,
        "partials_viewer/view.html",
        {"partial_path": partial_path, "content": content, "processed_content": processed_content},
    )
