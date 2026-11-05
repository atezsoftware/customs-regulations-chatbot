import { readFileSync } from "node:fs";
import { join } from "node:path";
import ts from "typescript";

const source = ts.createSourceFile(
  "AppPage.tsx",
  readFileSync(join(__dirname, "AppPage.tsx"), "utf8"),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX
);

function loadSubmitHandler(name: string, context: Record<string, unknown>) {
  let callback: ts.Expression | undefined;
  function visit(node: ts.Node) {
    if (
      ts.isVariableDeclaration(node) &&
      node.name.getText(source) === name &&
      node.initializer &&
      ts.isCallExpression(node.initializer)
    ) {
      callback = node.initializer.arguments[0];
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
  if (!callback) throw new Error(`Missing submit handler: ${name}`);
  const compiled = ts.transpileModule(
    `const handler = ${callback.getText(source)};`,
    { compilerOptions: { target: ts.ScriptTarget.ES2022 } }
  ).outputText;
  return new Function(...Object.keys(context), `${compiled}\nreturn handler;`)(
    ...Object.values(context)
  ) as (message: string) => Promise<void> | void;
}

function submitContext(workflow: string, multiModelActive = true) {
  const flags = Object.fromEntries(
    [
      "deepResearch",
      "atezSearch",
      "atezSearchV2",
      "atezSearchV3",
      "legalComposite",
      "legalReview",
      "supersearch",
      "asv3ExpCropped",
      "asv3Fast",
      "asv3FastCropped",
      "asv3Kompozit",
      "asv3KompozitCropped",
      "experimentalResearch",
      "experimentalParallelResearch",
      "experimentalGuardrails",
      "experimentalGuardrailsV2",
      "experimentalGuardrailsV3",
    ].map((name) => [`${name}EnabledForCurrentWorkflow`, name === workflow])
  );
  const selectedModels = [{ modelName: "first" }, { modelName: "second" }];
  return {
    ...flags,
    multiModel: { isMultiModelActive: multiModelActive, selectedModels },
    toast: { error: jest.fn() },
    resetInputBar: jest.fn(),
    onSubmit: jest.fn(),
    foldSidebarForMultiModel: jest.fn(),
    currentChatSessionId: "owned-session",
    currentMessageFiles: [{ id: "session-file", is_session_attachment: true }],
    messageHistory: [{ type: "user", message: "question", messageId: 1 }],
    showOnboarding: false,
    onboardingDismissed: true,
    finishOnboarding: jest.fn(),
  };
}

describe.each([
  "handleResubmitLastMessage",
  "onChat",
  "handleAppInputBarSubmit",
])("%s", (handlerName) => {
  it.each([
    ["atezSearchV3", true],
    ["asv3ExpCropped", true],
    ["atezSearchV3", false],
    ["asv3ExpCropped", false],
  ] as const)(
    "preserves model choices instead of downgrading %s to a generic request (draft files %s)",
    async (workflow, hasDraftFiles) => {
      const context = submitContext(workflow);
      if (!hasDraftFiles) context.currentMessageFiles = [];
      const handler = loadSubmitHandler(handlerName, context);

      await handler("question");

      expect(context.onSubmit).not.toHaveBeenCalled();
      expect(context.resetInputBar).not.toHaveBeenCalled();
      expect(context.toast.error).toHaveBeenCalledWith(
        "Dosyalı ASv3 sorgusu için tek model seçin."
      );
      expect(context.multiModel.selectedModels).toEqual([
        { modelName: "first" },
        { modelName: "second" },
      ]);
    }
  );

  it.each(["atezSearchV3", "asv3ExpCropped"])(
    "continues submitting %s with one selected model",
    async (workflow) => {
      const context = submitContext(workflow, false);
      await loadSubmitHandler(handlerName, context)("question");
      expect(context.onSubmit).toHaveBeenCalledTimes(1);
      expect(context.onSubmit).toHaveBeenCalledWith(
        expect.objectContaining({ [workflow]: true })
      );
    }
  );

  it("preserves model comparisons for other workflows", async () => {
    const context = submitContext("normal");
    await loadSubmitHandler(handlerName, context)("question");
    expect(context.onSubmit).toHaveBeenCalledTimes(1);
    expect(context.toast.error).not.toHaveBeenCalled();
  });
});
