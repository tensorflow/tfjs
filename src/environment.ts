/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

// Expects flags from URL in the format ?tfjsflags=FLAG1:1,FLAG2:true.
const TENSORFLOWJS_FLAGS_PREFIX = 'tfjsflags';

type FlagValue = number|boolean;
export type Flags = {
  [featureName: string]: FlagValue
};
export type FlagRegistryEntry = {
  evaluationFn: () => FlagValue;
  setHook?: (value: FlagValue) => void;
};

export class Environment {
  private flags: Flags = {};
  private flagRegistry: {[flagName: string]: FlagRegistryEntry} = {};

  private urlFlags: Flags = {};

  // tslint:disable-next-line: no-any
  constructor(public global: any) {
    this.populateURLFlags();
  }

  registerFlag(
      flagName: string, evaluationFn: () => FlagValue,
      setHook?: (value: FlagValue) => void) {
    this.flagRegistry[flagName] = {evaluationFn, setHook};

    // Override the flag value from the URL. This has to happen here because the
    // environment is initialized before flags get registered.
    if (this.urlFlags[flagName] != null) {
      const flagValue = this.urlFlags[flagName];
      console.warn(
          `Setting feature override from URL ${flagName}: ${flagValue}.`);
      this.set(flagName, flagValue);
    }
  }

  get(flagName: string): FlagValue {
    if (flagName in this.flags) {
      return this.flags[flagName];
    }

    this.flags[flagName] = this.evaluateFlag(flagName);

    return this.flags[flagName];
  }

  getNumber(flagName: string): number {
    return this.get(flagName) as number;
  }

  getBool(flagName: string): boolean {
    return this.get(flagName) as boolean;
  }

  getFlags(): Flags {
    return this.flags;
  }
  // For backwards compatibility.
  get features(): Flags {
    return this.flags;
  }

  set(flagName: string, value: FlagValue): void {
    if (this.flagRegistry[flagName] == null) {
      throw new Error(
          `Cannot set flag ${flagName} as it has not been registered.`);
    }
    this.flags[flagName] = value;
    if (this.flagRegistry[flagName].setHook != null) {
      this.flagRegistry[flagName].setHook(value);
    }
  }

  private evaluateFlag(flagName: string): FlagValue {
    if (this.flagRegistry[flagName] == null) {
      throw new Error(
          `Cannot evaluate flag '${flagName}': no evaluation function found.`);
    }
    return this.flagRegistry[flagName].evaluationFn();
  }

  setFlags(flags: Flags) {
    this.flags = Object.assign({}, flags);
  }

  reset() {
    this.flags = {};
    this.urlFlags = {};
    this.populateURLFlags();
  }

  private populateURLFlags(): void {
    if (typeof this.global === 'undefined' ||
        typeof this.global.location === 'undefined' ||
        typeof this.global.location.search === 'undefined') {
      return;
    }

    const urlParams = getQueryParams(this.global.location.search);
    if (TENSORFLOWJS_FLAGS_PREFIX in urlParams) {
      const keyValues = urlParams[TENSORFLOWJS_FLAGS_PREFIX].split(',');
      keyValues.forEach(keyValue => {
        const [key, value] = keyValue.split(':') as [string, string];
        this.urlFlags[key] = parseValue(key, value);
      });
    }
  }
}

export function getQueryParams(queryString: string): {[key: string]: string} {
  const params = {};
  queryString.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g, (s, ...t) => {
    decodeParam(params, t[0], t[1]);
    return t.join('=');
  });
  return params;
}

function decodeParam(
    params: {[key: string]: string}, name: string, value?: string) {
  params[decodeURIComponent(name)] = decodeURIComponent(value || '');
}

function parseValue(flagName: string, value: string): FlagValue {
  value = value.toLowerCase();
  if (value === 'true' || value === 'false') {
    return Boolean(value) === true;
  } else if (`${+ value}` === value) {
    return +value;
  }
  throw new Error(
      `Could not parse value flag value ${value} for flag ${flagName}.`);
}

export let ENV: Environment;
// tslint:disable-next-line:no-any
export function setEnvironmentGlobal(environment: Environment) {
  ENV = environment;
}
