const {execFile} = require('child_process');


function exec(command, args) {
  return new Promise((fulfill, reject) => {
    execFile(command, args, (error, stdout) => {
      if (error) {
        reject(error);
      } else {
        fulfill(stdout);
      }
    });
  });
}

describe('benchmark app cli', () => {
  beforeAll(() => {
    // Set a longer jasmine timeout than 5 seconds
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;
  });

  it('runs a command line command', async () => {
    expect(await exec('echo', ['-n', 'Hello, world!']))
        .toMatch('Hello');
  });

  it('runs a command that takes a while', async () => {
    expect(await exec('sleep', ['0.1'])).toEqual('');
  });

  it('runs a command that fails', async () => {
    await expectAsync(exec('exit 1')).toBeRejected();
  });
});
