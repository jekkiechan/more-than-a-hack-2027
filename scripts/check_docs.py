"""Check repository Markdown links, headings and directly routed skill references.

Uses only the standard library; reads Git-visible Markdown, never ignored data.
This checks the inline/reference links used here, not arbitrary HTML/Markdown.
Run from any directory: python /path/to/repo/scripts/check_docs.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r'\]\((<[^>]+>|[^\s)]+)(?:\s+"[^"]*")?\)')
DEFINITION = re.compile(r'^\s{0,3}\[[^\]]+\]:\s*(<[^>]+>|\S+)')
SCHEME = re.compile(r'^[a-zA-Z][a-zA-Z0-9+.-]*:|^//')


def prose_lines(text):
    """Yield numbered non-fenced lines, ignoring example links inside code."""
    fence = None
    for number, line in enumerate(text.splitlines(), 1):
        match = re.match(r'^\s{0,3}(`{3,}|~{3,})', line)
        if match:
            marker = match[1]
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            continue
        if fence is None:
            yield number, line


def headings(text):
    result, counts = set(), {}
    for _, line in prose_lines(text):
        match = re.match(r'^#{1,6}\s+(.+?)\s*#*$', line)
        if not match:
            continue
        title = re.sub(r'\[([^]]+)\]\([^)]+\)', r'\1', match[1])
        slug = re.sub(r'[^\w -]', '', title.lower()).replace(' ', '-')
        suffix = counts.get(slug, 0)
        counts[slug] = suffix + 1
        result.add(f'{slug}-{suffix}' if suffix else slug)
    return result


def links(text):
    for number, line in prose_lines(text):
        for match in LINK.finditer(line):
            yield number, match[1].strip('<>')
        definition = DEFINITION.match(line)
        if definition:
            yield number, definition[1].strip('<>')


def check_document(path, root):
    errors, count = [], 0
    for number, target in links(path.read_text()):
        if SCHEME.match(target):
            continue
        file, _, anchor = unquote(target).partition('#')
        destination = (path.parent / file).resolve() if file else path
        count += 1
        label = f'{path.relative_to(root)}:{number}'
        if not destination.exists():
            errors.append(f'{label}: missing target {target}')
        elif anchor and destination.suffix.lower() == '.md':
            if anchor not in headings(destination.read_text()):
                errors.append(f'{label}: missing heading {target}')
    return count, errors


def check_skill(skill, root):
    text = skill.read_text()
    errors = []
    front = re.match(r'\A---\nname: ([^\n]+)\ndescription: ([^\n]+)\n---\n', text)
    if not front or front[1] != skill.parent.name or not front[2].strip():
        errors.append(f'{skill.relative_to(root)}: invalid skill name/description frontmatter')
    targets = {(skill.parent / unquote(t).split('#')[0]).resolve()
               for _, t in links(text) if not SCHEME.match(t)}
    for reference in (skill.parent / 'references').glob('*.md'):
        if reference.resolve() not in targets:
            errors.append(f'{reference.relative_to(root)}: reference is not linked directly from SKILL.md')
    return errors


def main():
    output = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT)
    files = sorted({ROOT / p.decode() for p in output.split(b'\0') if p})
    count, errors, docs = 0, [], 0
    for path in files:
        if path.suffix.lower() != '.md' or not path.is_file():
            continue
        docs += 1
        n, found = check_document(path, ROOT)
        count += n
        errors.extend(found)
        if path.name == 'SKILL.md' and path.is_relative_to(ROOT / '.agents/skills'):
            errors.extend(check_skill(path, ROOT))
    for error in errors:
        print(error)
    print(f'{docs} Markdown files; {count} local links/anchors; {len(errors)} errors')
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
