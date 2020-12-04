// import * as argparse from 'argparse';
import {Project, SourceFile, SyntaxKind} from 'ts-morph';

// const parser = new argparse.ArgumentParser();

// initialize
const project = new Project({});

// interface OpInfo {
//   variableStatement: VariableStatement;
//   opFuncName: string;
//   opIdentifier: string;
//   opFunc: FunctionDeclaration;
// }

function convertToRunKernel(sourceFile: SourceFile) {
  // Find an invocation of Engine.runKernelFunc.
  // If none is found return false

  const descendantCallExpressions =
      sourceFile.getDescendantsOfKind(SyntaxKind.CallExpression);
  const runKernelFuncCalls = descendantCallExpressions.filter((callExpr) => {
    // console.log(callExpr.getFullText());
    return callExpr.getText().match(/^ENGINE.runKernelFunc/);
  });

  if (runKernelFuncCalls.length === 0) {
    return false;
  } else {
    console.log(`Found runKernelFunc in ${sourceFile.getFilePath()}`);
    runKernelFuncCalls.forEach(c => {
      const callArgs = c.getArguments();
      console.log(c.getText());

      // we want the 2nd, 4th and 5th (if present) arguments to construct our
      // new call.
      const [forwardFuncArg, inputsArg, gradArg, kernelNameArg, attrsArg] =
          callArgs;

      const kernelNameArgText = kernelNameArg.getText();
      const inputsArgText = inputsArg.getText();
      const attrsArgText = attrsArg == null ? '' : attrsArg.getText();

      // remove forwardFunc and grad from the call then change it to runKernel
      c.removeArgument(inputsArg);
      c.removeArgument(kernelNameArg);
      c.removeArgument(forwardFuncArg);
      c.removeArgument(gradArg);
      if (attrsArg != null) {
        c.removeArgument(attrsArg);
      }
      c.getChildAtIndex(0).replaceWithText('ENGINE.runKernel');

      // put the arguments back in the correct order
      c.addArgument(kernelNameArgText);
      c.addArgument(inputsArgText);
      if (attrsArgText.length > 0) {
        c.addArgument(attrsArgText);
      }

      // console.log('Modifed');
      // console.log(c.getText());
      // console.log('\n\n');
    });
    return true;
  }
}

async function run() {
  project.addSourceFilesAtPaths('src/ops/**/*.ts');
  const sourceFiles = project.getSourceFiles();

  await sourceFiles.forEach(async (opFile) => {
    if (!opFile.getFilePath().match('_test.ts')) {
      const found = convertToRunKernel(opFile);
      if (found) {
        opFile.fixUnusedIdentifiers();
        await opFile.save();
        opFile.fixUnusedIdentifiers();
        await opFile.save();
      }
    }
  });
}

// add source files
run();
