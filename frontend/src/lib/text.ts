/** Decode model-generated HTML entities before React renders plain chat text. */
export function decodeHtmlEntities(value: string): string {
  const element = document.createElement('textarea');
  element.innerHTML = value;
  return element.value;
}
