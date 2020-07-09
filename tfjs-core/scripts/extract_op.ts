import * as argparse from 'argparse';
import * as path from 'path';
import {FunctionDeclaration, ImportDeclaration, Project, SourceFile, VariableStatement} from 'ts-morph';

const parser = new argparse.ArgumentParser();

parser.addArgument(
    '--op_file',
    {help: 'path to op file. e.g. src/ops/unary_ops.ts', required: true});
parser.addArgument('--ops', {
  help: 'comma seprated list of ops to extract (e.g. tanh,tan).' +
      ' Skip this param to extract all ops from the file.',
  defaultValue: [],
});

// initialize
const project = new Project({});

interface OpInfo {
  variableStatement: VariableStatement;
  opFuncName: string;
  opIdentifier: string;
  opFunc: FunctionDeclaration;
}

function getImports(sourceFile: SourceFile) {
  return sourceFile.getImportDeclarations();
}

function getOpExports(file: SourceFile): OpInfo[] {
  const variables = file.getVariableStatements();
  const opFuncRegex = /op\(\{(.*_)\}\)/;
  const exported = variables.filter(
      v => v.isExported() && v.getFullText().match(opFuncRegex));
  const opInfo = exported.map(variable => {
    const declaration = variable.getDeclarations()[0];
    const opFuncName = variable.getFullText().match(opFuncRegex)[1];
    const opFunc = getOpFunc(file, opFuncName);
    if (opFunc == null) {
      console.warn(`Warning: could not find implementation function for ${
          declaration.getName()}`);
      return null;
    }
    return {
      variableStatement: variable,
      // string with exported name of op
      opIdentifier: declaration.getName(),
      // string with name of the function that actually implements the op
      opFuncName,
      // function that implements the op
      opFunc,
    };
  });
  return opInfo.filter(op => op != null);
}

function getOpFunc(sourceFile: SourceFile, opFuncName: string) {
  return sourceFile.getFunction(opFuncName);
}

function toSnakeCase(str: string) {
  // add exceptions here.
  if (str === 'isNaN') {
    return 'is_nan';
  }
  return str.replace(/[A-Z]/g, (s: string) => `_${s.toLowerCase()}`);
}

async function moveToNewFile(
    opInfo: OpInfo, imports: ImportDeclaration[], sourceFile: SourceFile) {
  //
  // Move code to a new file
  //
  const targetFp = `src/ops/${toSnakeCase(opInfo.opIdentifier)}.ts`;
  const newOpFile = project.createSourceFile(targetFp, (writer) => {
    // By using getFullText here we will also get the copyright notice at the
    // begining of the file
    const importsStr = imports.map(i => i.getFullText()).join('');
    const functionStr = opInfo.opFunc.getFullText();
    const exportString = opInfo.variableStatement.getFullText();
    const contents = [importsStr, functionStr, exportString].join('');
    writer.write(contents);
  }, {overwrite: true});
  newOpFile.fixUnusedIdentifiers();
  await newOpFile.save();

  // Add export to ops file.
  const opsExportFile = project.getSourceFile('src/ops/ops.ts');
  opsExportFile.addExportDeclaration({
    namedExports: [opInfo.opIdentifier],
    moduleSpecifier: `./${path.basename(targetFp, '.ts')}`,
  });

  await opsExportFile.save();
}

async function run(filePath: string, ops: string[]) {
  console.log('ops', ops);
  project.addSourceFilesAtPaths(filePath);
  // add the ops export file to the project
  project.addSourceFilesAtPaths('src/ops/ops.ts');
  const opFile = project.getSourceFile(filePath);
  const imports = getImports(opFile);
  const opExports = getOpExports(opFile);

  opExports.forEach(async o => {
    if (ops.length === 0 || ops.indexOf(o.opIdentifier) !== -1) {
      await moveToNewFile(o, imports, opFile);
    }
  });

  // Remove the op from the source file and save it
  opExports.forEach(async o => {
    if (ops.length === 0 || ops.indexOf(o.opIdentifier) !== -1) {
      opFile.removeStatement(o.variableStatement.getChildIndex());
    }
  });
  opFile.fixUnusedIdentifiers();
  await opFile.save();
}

// add source files

const args = parser.parseArgs();
let opsToExtract = args.ops;
if (!Array.isArray(opsToExtract)) {
  opsToExtract = opsToExtract.split(',');
}
console.log('Extracting from', args.op_file);
if (opsToExtract.length > 0) {
  console.log('Only extract: ', opsToExtract);
}

run(args.op_file, opsToExtract);
