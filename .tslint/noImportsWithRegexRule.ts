import * as Lint from 'tslint';
import * as ts from 'typescript';

export class Rule extends Lint.Rules.AbstractRule {
  public apply(sourceFile: ts.SourceFile): Lint.RuleFailure[] {
    return this.applyWithWalker(
        new NoImportsWithRegexWalker(sourceFile, this.getOptions()));
  }
}

class NoImportsWithRegexWalker extends Lint.RuleWalker {
  public visitImportDeclaration(node: ts.ImportDeclaration) {
    const options = this.getOptions();
    for (const regStr of options) {
      const importFrom = node.moduleSpecifier.getText();
      const reg = new RegExp(regStr);
      if (importFrom.match(reg)) {
        this.addFailure(this.createFailure(
            node.moduleSpecifier.getStart(), node.moduleSpecifier.getWidth(),
            `importing from ${regStr} is prohibited.`));
      }
    }

    super.visitImportDeclaration(node);
  }
}
