import importlib.resources
import os
import warnings

import markdown
import pdfkit

from openavmkit.utilities.settings import get_model_group, get_valuation_date


class MarkdownReport:
  """
  A report generator that uses a Markdown template.

  Attributes:
      name (str): Name of the report, corresponding to a Markdown template.
      template (str): The raw Markdown template text.
      rendered (str): The rendered Markdown text after variable substitution.
      variables (dict): Dictionary of variables for substitution in the template.
  """

  name: str
  template: str
  rendered: str
  variables: dict

  def __init__(self, name):
    """
    Initialize the MarkdownReport by loading the Markdown template.

    :param name: Name of the report template (without file extension).
    :type name: str
    """
    self.name = name
    with importlib.resources.open_text("openavmkit.resources.reports", f"{name}.md", encoding="utf-8") as file:
      self.template = file.read()
    self.variables = {}
    self.rendered = ""

  def get_var(self, key: str):
    """
    Get the value of a variable.

    :param key: Variable key.
    :type key: str
    :returns: The value associated with the key, or None if not set.
    """
    return self.variables.get(key)

  def set_var(self, key: str, value, fmt: str = None):
    """
    Set a variable value with optional formatting.

    :param key: Variable key.
    :type key: str
    :param value: Value to be set.
    :type value: any
    :param fmt: Optional format string.
    :type fmt: str, optional
    """
    if value is None:
      formatted_value = "<NULL>"
    elif fmt is not None:
      formatted_value = format(value, fmt)
    else:
      formatted_value = str(value)
    self.variables[key] = formatted_value

  def render(self):
    """
    Render the report by substituting variables in the template.

    :returns: Rendered Markdown text.
    :rtype: str
    """
    self.rendered = self.template
    for key in self.variables:
      value = self.variables.get(key)
      self.rendered = self.rendered.replace("{{{" + key + "}}}", str(value))
    return self.rendered


def start_report(report_name: str, settings: dict, model_group: str):
  """
  Create and initialize a MarkdownReport with basic variables set.

  :param report_name: Name of the report template.
  :type report_name: str
  :param settings: Settings dictionary.
  :type settings: dict
  :param model_group: Model group identifier.
  :type model_group: str
  :returns: Initialized MarkdownReport object.
  :rtype: MarkdownReport
  """
  report = MarkdownReport(report_name)
  locality = settings.get("locality", {}).get("name")
  val_date = get_valuation_date(settings)
  val_date = val_date.strftime("%Y-%m-%d")

  model_group_obj = get_model_group(settings, model_group)
  model_group_name = model_group_obj.get("name", model_group)

  report.set_var("locality", locality)
  report.set_var("val_date", val_date)
  report.set_var("model_group", model_group_name)
  return report


def finish_report(report: MarkdownReport, outpath: str, css_file: str, settings: dict):
  """
  Render the report and export it in Markdown, HTML and PDF formats, as specified.

  Saves the rendered Markdown to disk and converts it to target formats using a specified CSS file.


  :param report: MarkdownReport object to be finished.
  :type report: MarkdownReport
  :param outpath: Output file path (without extension).
  :type outpath: str
  :param css_file: Name of the CSS file (without extension) to style the report.
  :type css_file: str
  :param settings: Settings dictionary.
  :type settings: dict
  """
  formats = settings.get("analysis", {}).get("report", {}).get("formats", None)
  if formats is None:
    formats = ["pdf", "md"]

  report_text = report.render()
  os.makedirs(outpath, exist_ok=True)
  with open(f"{outpath}.md", "w", encoding="utf-8") as f:
    f.write(report_text)
  pdf_path = f"{outpath}.pdf"

  _markdown_to_pdf(report_text, pdf_path, formats=formats, css_file=css_file)

  if "md" not in formats:
    os.remove(f"{outpath}.md")


def _markdown_to_pdf(md_text, out_path, formats, css_file=None):
  """
  Convert Markdown text to a PDF file.

  First converts Markdown to HTML, saves the HTML to disk, then converts the HTML to PDF.

  :param md_text: Markdown text.
  :type md_text: str
  :param out_path: Output PDF file path.
  :type out_path: str
  :param formats: List of formats to output (e.g., ["pdf", "md", "html"]).
  :type formats: list
  :param css_file: Optional CSS file stub for styling.
  :type css_file: str, optional
  """
  html_text = _markdown_to_html(md_text, css_file)
  html_path = out_path.replace(".pdf", ".html")
  with open(html_path, "w", encoding="utf-8") as html_file:
    html_file.write(html_text)

  if "pdf" in formats:
    _html_to_pdf(html_text, out_path)

  if "html" not in formats:
    os.remove(html_path)


def _markdown_to_html(md_text, css_file_stub=None):
  """
  Convert Markdown text to a complete HTML document using a CSS file.

  :param md_text: Markdown text.
  :type md_text: str
  :param css_file_stub: Optional CSS file stub (without extension).
  :type css_file_stub: str, optional
  :returns: HTML text.
  :rtype: str
  """
  html_text = markdown.markdown(md_text, extensions=["extra"])

  css_path = _get_resource_path() + f"/reports/css/{css_file_stub}.css"

  if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as css_file:
      css_text = css_file.read()
  else:
    css_text = ""

  css_base_path = _get_resource_path() + f"/reports/css/base.css"
  with open(css_base_path, "r", encoding="utf-8") as css_file:
    css_base_text = css_file.read()

  css_text = css_base_text + css_text
  html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
        {css_text}
        </style>
    </head>
    <body>
        {html_text}
    </body>
    </html>
    """
  return html_template


def _html_to_pdf(html_text, out_path):
  """
  Convert an HTML string to a PDF file using pdfkit.

  :param html_text: HTML content.
  :type html_text: str
  :param out_path: Output PDF file path.
  :type out_path: str
  """
  try:
    pdfkit.from_string(html_text, out_path, options={"quiet": False})
  except OSError:
    warnings.warn("Failed to generate PDF report. Is `wkhtmltopdf` installed? See the README for details.")


def _get_resource_path():
  """
  Get the absolute path to the resources directory.

  :returns: Path to the resources directory.
  :rtype: str
  """
  this_files_path = os.path.abspath(__file__)
  this_files_dir = os.path.dirname(this_files_path)
  resources_path = os.path.join(this_files_dir, "resources")
  return resources_path
