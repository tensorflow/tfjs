import * as Lint from 'tslint';
import * as ts from 'typescript';

export class Rule extends Lint.Rules.AbstractRule {
  public static FAILURE_STRING =
      'importing from src/ is prohibited. You should be able to import from a closer path';

  public apply(sourceFile: ts.SourceFile): Lint.RuleFailure[] {
    return this.applyWithWalker(
        new NoImportsFromSrcWalker(sourceFile, this.getOptions()));
  }
}

class NoImportsFromSrcWalker extends Lint.RuleWalker {
  public visitImportDeclaration(node: ts.ImportDeclaration) {
    const importFrom = node.moduleSpecifier.getText();
    const reg = /src/;
    if (importFrom.match(reg)) {
      this.addFailure(this.createFailure(
          node.moduleSpecifier.getStart(), node.moduleSpecifier.getWidth(),
          Rule.FAILURE_STRING));
    }

    super.visitImportDeclaration(node);
  }
}
