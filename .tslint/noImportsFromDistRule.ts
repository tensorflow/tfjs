import * as Lint from "tslint";
import * as ts from "typescript";

export class Rule extends Lint.Rules.AbstractRule {
  public static FAILURE_STRING = "importing from dist/ is prohibited. Please use public API";

  public apply(sourceFile: ts.SourceFile): Lint.RuleFailure[] {
    return this.applyWithWalker(
        new NoImportsFromDistWalker(sourceFile, this.getOptions()));
  }
}

class NoImportsFromDistWalker extends Lint.RuleWalker {
  public visitImportDeclaration(node: ts.ImportDeclaration) {
    const importFrom = node.moduleSpecifier.getText();
    const reg = /@tensorflow\/tfjs[-a-z]*\/dist/;
    if (importFrom.match(reg)) {
      const fix = new Lint.Replacement(node.getStart(), node.getWidth(),
          "import * as tf from '@tensorflow/tfjs-core';");

      this.addFailure(this.createFailure(node.getStart(),
          node.getWidth(), Rule.FAILURE_STRING, fix));
    }

    super.visitImportDeclaration(node);
  }

}
