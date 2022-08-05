class PromiseQueue {
  queue = [];
  running = 0;

  constructor(concurrency = Infinity) {
    this.concurrency = concurrency;
  }

  add(func) {
    return new Promise((resolve, reject) => {
      this.queue.push(async () => {
	this.running++;
	try {
	  resolve(await func());
	} finally {
	  this.running--;
	  this.run();
	}
      });
      this.run();
    });
  }

  run() {
    while (this.running < this.concurrency && this.queue.length > 0) {
      this.queue.shift()();
    }
  }
}

module.exports = {PromiseQueue};
