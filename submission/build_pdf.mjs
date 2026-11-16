// Render submission/TECHNICAL_DOCUMENT.md to HTML (pandoc + KaTeX + Mermaid),
// then print it to PDF with Playwright's Chromium.
// Usage: node submission/build_pdf.mjs ["Douki Douki"]
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const team = process.argv[2] ?? 'Douki Douki';
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const src = path.join(root, 'submission', 'TECHNICAL_DOCUMENT.md');
const outDir = path.join(root, 'data', 'submission');
mkdirSync(outDir, { recursive: true });
const repo = 'https://github.com/jekkiechan/more-than-a-hack-2027/blob/main';

let body = execFileSync('pandoc', [src, '-f', 'gfm+tex_math_dollars', '-t', 'html5', '--katex'], {
  encoding: 'utf8',
});
// Mermaid blocks become client-rendered diagrams.
body = body.replace(/<pre class="mermaid"><code>([\s\S]*?)<\/code><\/pre>/g, '<div class="mermaid">$1</div>');
// Relative repository links become absolute so they work inside the PDF.
body = body
  .replace(/href="\.\.\/([^"]+)"/g, `href="${repo}/$1"`)
  .replace(/href="(README\.md[^"]*)"/g, `href="${repo}/submission/$1"`);

const html = `<!doctype html><html><head><meta charset="utf-8">
<title>${team} technical document</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"></script>
<style>
  @page { size: A4; margin: 15mm 16mm; }
  body { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 9.6pt; line-height: 1.38; color: #1a1a1a; }
  h1 { font-size: 16pt; margin: 0 0 2pt; }
  h1 + p { color: #555; margin-top: 0; }
  h2 { font-size: 11.5pt; margin: 8pt 0 3pt; border-bottom: 1px solid #ccc; padding-bottom: 1pt; }
  p { margin: 3pt 0 5pt; }
  ul, ol { margin: 2pt 0 5pt; padding-left: 16pt; }
  li { margin: 1pt 0; }
  table { border-collapse: collapse; margin: 4pt 0 6pt; width: 100%; font-size: 8.8pt; }
  th, td { border: 1px solid #d0d0d0; padding: 2pt 5pt; vertical-align: top; text-align: left; }
  th { background: #f2f2f2; }
  td[align="right"], th[align="right"] { text-align: right; }
  code { font-family: Menlo, Consolas, monospace; font-size: 8.4pt; background: #f4f4f4; padding: 0 2px; border-radius: 2px; }
  pre { background: #f4f4f4; padding: 4pt 6pt; margin: 3pt 0 5pt; white-space: pre-wrap; word-break: break-all; }
  pre code { background: none; padding: 0; }
  a { color: #1f5fa8; text-decoration: none; }
  .mermaid { text-align: center; margin: 4pt 0; }
  .mermaid svg { max-height: 165px; max-width: 100%; }
  .page-break { break-after: page; }
  .katex-display { margin: 4pt 0; }
  h2, h3 { break-after: avoid; }
  tr, li, .katex-display, .mermaid { break-inside: avoid; }
</style></head><body>${body}
<script>
  document.querySelectorAll('.math').forEach(el => {
    katex.render(el.textContent, el, { displayMode: el.classList.contains('display'), throwOnError: true });
  });
  mermaid.initialize({ startOnLoad: false, theme: 'neutral', flowchart: { htmlLabels: false } });
  mermaid.run().then(() => { document.body.dataset.ready = '1'; });
</script></body></html>`;

const htmlPath = path.join(outDir, `${team}.html`);
const pdfPath = path.join(outDir, `${team}.pdf`);
writeFileSync(htmlPath, html);

const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto('file://' + htmlPath);
await page.waitForSelector('body[data-ready="1"]', { timeout: 30000 });
await page.pdf({ path: pdfPath, format: 'A4', printBackground: true, preferCSSPageSize: true });
await browser.close();
console.log(pdfPath);
