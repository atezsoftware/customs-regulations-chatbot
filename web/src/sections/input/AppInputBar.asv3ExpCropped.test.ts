import { readFileSync } from "node:fs";
import { join } from "node:path";
import ts from "typescript";

it.each([["toggleAsv3ExpCropped", "asv3ExpCroppedEnabled", "exp cropped"]])(
  "shows %s as an independent single-model workflow selection",
  (toggle, enabled, label) => {
    const source = ts.createSourceFile(
      "AppInputBar.tsx",
      readFileSync(join(__dirname, "AppInputBar.tsx"), "utf8"),
      ts.ScriptTarget.Latest,
      true,
      ts.ScriptKind.TSX
    );
    const selections: ts.JsxElement[] = [];
    function visit(node: ts.Node) {
      if (
        ts.isJsxElement(node) &&
        node.openingElement.tagName.getText(source) === "SelectButton" &&
        node.openingElement.attributes.properties.some(
          (attribute) =>
            ts.isJsxAttribute(attribute) &&
            attribute.name.getText(source) === "onClick" &&
            attribute.initializer?.getText(source) === `{${toggle}}`
        )
      )
        selections.push(node);
      ts.forEachChild(node, visit);
    }
    visit(source);
    expect(selections).toHaveLength(1);
    const selection = selections[0]!;
    expect(
      selection.children
        .map((child) => child.getText(source))
        .join("")
        .trim()
    ).toBe(label);
    expect(selection.openingElement.getText(source)).toContain(
      `state={${enabled} ? "selected" : "empty"}`
    );
    expect(selection.openingElement.getText(source)).toContain(
      "disabled={disabled || isMultiModelActive}"
    );
  }
);
