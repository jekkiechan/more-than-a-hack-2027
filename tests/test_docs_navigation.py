"""Contracts for the documentation checker, using temporary Markdown only."""
import tempfile
import unittest
from pathlib import Path

from scripts.check_docs import check_document, check_skill, headings


class DocumentationNavigationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.doc = self.root / 'README.md'

    def check(self, text):
        self.doc.write_text(text)
        return check_document(self.doc, self.root)

    def test_missing_file_is_actionable(self):
        count, errors = self.check('[gone](missing.md)')
        self.assertEqual(count, 1)
        self.assertEqual(errors, ['README.md:1: missing target missing.md'])

    def test_heading_and_duplicate_slug(self):
        text = '# Same\n# Same\n[x](#same-1)\n'
        self.assertEqual(self.check(text), (1, []))
        self.assertEqual(headings(text), {'same', 'same-1'})

    def test_missing_heading_fails(self):
        self.assertIn('missing heading', self.check('# Present\n[x](#absent)')[1][0])

    def test_fenced_examples_and_remote_links_are_ignored(self):
        text = '```md\n[x](missing.md)\n# Wrong\n```\n[x](https://example.com)\n'
        self.assertEqual(self.check(text), (0, []))
        self.assertEqual(headings(text), set())

    def test_encoded_and_angle_paths(self):
        (self.root / 'A B.md').write_text('# Title\n')
        self.assertEqual(self.check('[x](A%20B.md#title)\n[y](<A B.md#title>)'), (2, []))

    def test_reference_definition_is_checked(self):
        count, errors = self.check('[x][doc]\n\n[doc]: missing.md\n')
        self.assertEqual(count, 1)
        self.assertIn('README.md:3: missing target', errors[0])

    def test_skill_reference_must_be_directly_routed(self):
        skill = self.root / 'example/SKILL.md'
        skill.parent.mkdir()
        (skill.parent / 'references').mkdir()
        ref = skill.parent / 'references/task.md'
        ref.write_text('# Task\n')
        skill.write_text('---\nname: example\ndescription: Do the task.\n---\n')
        self.assertEqual(len(check_skill(skill, self.root)), 1)
        skill.write_text(skill.read_text() + '[Task](references/task.md)\n')
        self.assertEqual(check_skill(skill, self.root), [])

    def test_skill_name_matches_its_directory(self):
        skill = self.root / 'example/SKILL.md'
        skill.parent.mkdir()
        skill.write_text('---\nname: wrong\ndescription: Do the task.\n---\n')
        self.assertIn('frontmatter', check_skill(skill, self.root)[0])


if __name__ == '__main__':
    unittest.main()
