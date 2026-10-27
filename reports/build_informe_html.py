"""
Genera una versión HTML autocontenida de ``reports/INFORME_PROYECTO.md``.

- Renderiza Markdown con mistune (tablas incluidas).
- Convierte las imágenes de ``figures/`` a data URI para que el HTML se pueda
  mover o imprimir sin depender de archivos externos.
- Añade ids a los encabezados para que funcionen los enlaces del índice.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path
from urllib.parse import unquote

import mistune

ROOT = Path(__file__).resolve().parent
MD = ROOT / "INFORME_PROYECTO.md"
HTML = ROOT / "INFORME_PROYECTO.html"


def slug(texto: str) -> str:
    """Slug compatible con los enlaces manuales del índice."""
    s = texto.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "-", s)
    return s.strip("-")


def agregar_ids(html: str) -> str:
    def repl(match: re.Match) -> str:
        nivel, texto = match.group(1), match.group(2)
        limpio = re.sub(r"<[^>]+>", "", texto)
        return f'<h{nivel} id="{slug(limpio)}">{texto}</h{nivel}>'

    return re.sub(r"<h([23])>(.*?)</h\1>", repl, html, flags=re.S)


def incrustar_imagenes(html: str) -> str:
    def repl(match: re.Match) -> str:
        ruta = ROOT / match.group(1)
        if not ruta.exists():
            return match.group(0)
        datos = base64.b64encode(ruta.read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{datos}"'

    return re.sub(r'src="(figures/[^"]+)"', repl, html)


def decodificar_enlaces(html: str) -> str:
    """Mistune codifica los acentos del índice; los ids no lo están."""
    return re.sub(
        r'href="#([^"]+)"',
        lambda m: f'href="#{unquote(m.group(1))}"',
        html,
    )


def main() -> None:
    texto = MD.read_text(encoding="utf-8")
    cuerpo = mistune.create_markdown(plugins=["table"])(texto)
    cuerpo = agregar_ids(cuerpo)
    cuerpo = incrustar_imagenes(cuerpo)
    cuerpo = decodificar_enlaces(cuerpo)

    html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Informe del proyecto · Sedimentos de Moquegua</title>
<style>
  :root {{
    --tinta: #17212b;
    --suave: #5b6b7c;
    --linea: #d9e1e8;
    --fondo: #ffffff;
    --acento: #0b6e99;
    --acento-claro: #eaf4f9;
    --alerta: #b03a2e;
  }}
  * {{ box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    margin: 0;
    background: #f4f6f8;
    color: var(--tinta);
    font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.65;
    font-size: 16px;
  }}
  main {{
    max-width: 1080px;
    margin: 28px auto 60px;
    background: var(--fondo);
    padding: 48px 56px 64px;
    box-shadow: 0 8px 30px rgba(20, 40, 60, .08);
    border-radius: 10px;
  }}
  h1, h2, h3, h4 {{ line-height: 1.25; color: #0e2233; }}
  h1 {{ font-size: 2.1rem; margin: 0 0 .4rem; letter-spacing: -.02em; }}
  h2 {{
    font-size: 1.55rem;
    margin: 2.6rem 0 .9rem;
    padding-top: .6rem;
    border-top: 3px solid var(--acento);
    page-break-after: avoid;
  }}
  h3 {{ font-size: 1.2rem; margin: 1.9rem 0 .6rem; color: #123b52; }}
  p {{ margin: .7rem 0; }}
  a {{ color: var(--acento); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  strong {{ color: #0b2a3d; }}
  blockquote {{
    margin: 1.2rem 0;
    padding: .9rem 1.1rem;
    border-left: 5px solid var(--acento);
    background: var(--acento-claro);
    color: #17384b;
    border-radius: 0 6px 6px 0;
  }}
  blockquote p {{ margin: .25rem 0; }}
  code {{
    font-family: "Cascadia Code", Consolas, monospace;
    font-size: .9em;
    background: #eef2f5;
    padding: .12em .35em;
    border-radius: 4px;
  }}
  pre {{
    background: #10202c;
    color: #e9f2f7;
    padding: 1rem 1.2rem;
    border-radius: 8px;
    overflow-x: auto;
    line-height: 1.45;
  }}
  pre code {{ background: transparent; color: inherit; padding: 0; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 1.1rem 0 1.5rem;
    font-size: .92rem;
    page-break-inside: avoid;
  }}
  th, td {{
    border: 1px solid var(--linea);
    padding: .48rem .6rem;
    vertical-align: top;
  }}
  th {{ background: #e8f1f6; text-align: left; color: #0e344a; }}
  tr:nth-child(even) td {{ background: #fafcfd; }}
  td:has(+ td), th:has(+ td) {{ }}
  img {{
    display: block;
    max-width: 100%;
    height: auto;
    margin: 1.4rem auto .35rem;
    border: 1px solid #e3e9ee;
    border-radius: 6px;
    background: #fff;
  }}
  img + em, p > em:only-child {{
    display: block;
    text-align: center;
    color: var(--suave);
    font-size: .88rem;
    margin: .2rem auto 1.6rem;
    max-width: 90%;
  }}
  hr {{ border: 0; border-top: 1px solid var(--linea); margin: 2.2rem 0; }}
  ul, ol {{ padding-left: 1.5rem; }}
  li {{ margin: .3rem 0; }}
  .aviso {{
    background: #fff7e6;
    border: 1px solid #f0d49a;
    padding: .8rem 1rem;
    border-radius: 6px;
  }}
  @media (max-width: 760px) {{
    main {{ margin: 0; padding: 24px 18px 40px; border-radius: 0; }}
    h1 {{ font-size: 1.6rem; }}
    h2 {{ font-size: 1.3rem; }}
    table {{ font-size: .82rem; }}
    th, td {{ padding: .35rem .4rem; }}
  }}
  @media print {{
    body {{ background: #fff; font-size: 11pt; }}
    main {{ box-shadow: none; margin: 0; max-width: none; padding: 0; }}
    h2 {{ page-break-after: avoid; }}
    h3 {{ page-break-after: avoid; }}
    table, img, blockquote, pre {{ page-break-inside: avoid; }}
    a {{ color: #000; text-decoration: none; }}
  }}
</style>
</head>
<body>
<main>
{cuerpo}
</main>
</body>
</html>
"""
    HTML.write_text(html, encoding="utf-8")
    print(f"HTML generado: {HTML}")
    print(f"Tamaño: {HTML.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
