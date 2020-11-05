import * as argparse from 'argparse';
import {ClassDeclaration, MethodDeclaration, Project, SourceFile} from 'ts-morph';

const parser = new argparse.ArgumentParser();
parser.addArgument('--kernels', {
  help: 'comma separated list of kernels to extract (e.g. tanh,tan).',
  defaultValue: [],
  required: true
});

// initialize
const project = new Project({});

function getKernelMethod(
    classDecl: ClassDeclaration, opFuncName: string): MethodDeclaration {
  return classDecl.getInstanceMethod(opFuncName);
}

export function getAttrs(
    kernelName: string, kernelNamesFile: SourceFile): string[] {
  const attrDecl = kernelNamesFile.getInterface(
      s => s.getText().includes(`interface ${kernelName}Attrs`));

  let propNames: string[] = [];
  if (attrDecl != null) {
    propNames =
        attrDecl.getType().getApparentProperties().map(t => t.getEscapedName());
  }
  return propNames;
}

function getInputs(kernelName: string, kernelNamesFile: SourceFile): string[] {
  let inputDecl;
  inputDecl = kernelNamesFile.getTypeAlias(
      s => s.getText().includes(`type ${kernelName}Inputs`));
  if (inputDecl == null) {
    inputDecl = kernelNamesFile.getInterface(
        s => s.getText().includes(`interface ${kernelName}Inputs`));
  }
  if (inputDecl == null) {
    // There are a small number of kernels that don't have inputs but
    // we can deal with those separately as the most likely issue is
    // an error.
    return [];
  }

  const propNames =
      inputDecl.getType().getApparentProperties().map(t => t.getEscapedName());
  return propNames;
}

function upcaseFirstChar(str: string) {
  if (str.startsWith('_')) {
    // e.g. _fusedMatMul
    return `_${str.charAt(1).toUpperCase()}${str.slice(2)}`;
  }
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function getPreamble() {
  const preamble = `/**
 * @license
 * Copyright ${(new Date()).getFullYear()} Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

`;
  return preamble;
}

function getImports(kernelName: string, attrs: string[]) {
  const kernelInfoImport = attrs.length > 0 ?
      `import {KernelConfig, ${kernelName}, ${kernelName}Inputs, ${
          kernelName}Attrs}` :
      `import {KernelConfig, ${kernelName}, ${kernelName}Inputs}`;
  const imports = `${kernelInfoImport} from '@tensorflow/tfjs';
import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';
`;
  return imports;
}

function getKernelConfigBody(
    kernelFuncName: string, kernelName: string, inputs: string[],
    attrs: string[], kernelFuncBody: string) {
  let inputDestructure =
      `const {${inputs.join(', ')}} = args.inputs as ${kernelName}Inputs;
\t\tconst backend = args.backend as NodeJSKernelBackend;`;
  if (attrs.length > 0) {
    inputDestructure += `\n\t\tconst {${
        attrs.join(', ')}} = args.attrs as {} as ${kernelName}Attrs;`;
  }

  const configBody = `
export const ${kernelFuncName}Config: KernelConfig = {
  kernelName: ${kernelName},
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    ${inputDestructure}

    ${kernelFuncBody}
  }
};
`;

  return configBody;
}

export async function moveToNewFile(
    kernelFuncName: string, kernelFunc: MethodDeclaration, inputs: string[],
    attrs: string[]) {
  const kernelName = upcaseFirstChar(kernelFuncName);
  const functionBody = kernelFunc.getBodyText()
                           .replace(/this/g, 'backend')
                           .replace(new RegExp(`'${kernelName}'`), kernelName);

  //
  // Move code to a new file
  //

  const targetFp = `src/kernels/${kernelName}.ts`;

  const newKernelFile = project.createSourceFile(targetFp, (writer) => {
    const preamble = getPreamble();
    const importStatments = getImports(kernelName, attrs);
    const kernelConfigBody = getKernelConfigBody(
        kernelFuncName, kernelName, inputs, attrs, functionBody);

    const contents = [preamble, importStatments, kernelConfigBody].join('');
    writer.write(contents);
  }, {overwrite: true});

  // newKernelFile.fixUnusedIdentifiers();
  await newKernelFile.save();
}

function registerKernel(
    kernelFuncName: string, kernelName: string,
    registerkernelsFile: SourceFile) {
  const configVarName = `${kernelFuncName}Config`;

  registerkernelsFile.addImportDeclaration({
    namedImports: [configVarName],
    moduleSpecifier: `./kernels/${kernelName}`,
  });

  // get the kernelConfig var decl
  const configsVar =
      registerkernelsFile.getVariableDeclaration('kernelConfigs');

  const arrayChildPos = 4;
  const array = configsVar.getChildAtIndex(arrayChildPos);
  const configStrs = array.getChildAtIndex(1)
                         .getChildren()
                         .map(c => c.getText())
                         .filter(s => s.length > 1);

  configStrs.push(configVarName);
  configStrs.sort();

  const contents =
      [`const kernelConfigs: KernelConfig[] = [`, configStrs.join(',\n'), `];`];

  configsVar.getVariableStatement().replaceWithText(contents.join('\n'));
  registerkernelsFile.saveSync();
}

async function run(kernelFuncNames: string[]) {
  const backendFilePath = 'src/nodejs_kernel_backend.ts';
  const kernelNamesFilePath = '../tfjs-core/src/kernel_names.ts';
  const registerkernelsFilePath = 'src/register_all_kernels.ts';
  project.addSourceFilesAtPaths(backendFilePath);
  project.addSourceFilesAtPaths(kernelNamesFilePath);
  project.addSourceFilesAtPaths(registerkernelsFilePath);

  const backendFile = project.getSourceFile(backendFilePath);
  const kernelNamesFile = project.getSourceFile(kernelNamesFilePath);
  const registerkernelsFile = project.getSourceFile(registerkernelsFilePath);

  const backendClass = backendFile.getClass('NodeJSKernelBackend');

  kernelFuncNames.forEach(async (kernelFuncName) => {
    try {
      const kernelName = upcaseFirstChar(kernelFuncName);
      // Get the kernel func definition in the backend class.
      const func = getKernelMethod(backendClass, kernelFuncName);
      const inputs = getInputs(kernelName, kernelNamesFile);
      const attrs = getAttrs(kernelName, kernelNamesFile);

      console.log('Found func', func.getBodyText());
      console.log('Found inputs', inputs);
      console.log('Found attrs', attrs);
      // Move this definition to a new file.
      await moveToNewFile(kernelFuncName, func, inputs, attrs);
    } catch (e) {
      console.log(e);
      process.exit();
    }
  });

  // register the new kernels
  kernelFuncNames.forEach((kernelFuncName) => {
    const kernelName = upcaseFirstChar(kernelFuncName);
    registerKernel(kernelFuncName, kernelName, registerkernelsFile);
  });

  await registerkernelsFile.save();

  kernelFuncNames.forEach((kernelFuncName) => {
    // Delete the kernel func definition from the backend class
    const func = getKernelMethod(backendClass, kernelFuncName);
    func.remove();
  });
  await backendFile.save();
}

// add source files

const args = parser.parseArgs();
let kernelsToExtract = args.kernels;
if (!Array.isArray(kernelsToExtract)) {
  kernelsToExtract = kernelsToExtract.split(',');
}
if (kernelsToExtract.length > 0) {
  console.log('Only extract: ', kernelsToExtract);
}

run(kernelsToExtract);
