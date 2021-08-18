"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
exports.__esModule = true;
var Lint = require("tslint");
var Rule = /** @class */ (function (_super) {
    __extends(Rule, _super);
    function Rule() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Rule.prototype.apply = function (sourceFile) {
        return this.applyWithWalker(new NoImportsWithRegexWalker(sourceFile, this.getOptions()));
    };
    return Rule;
}(Lint.Rules.AbstractRule));
exports.Rule = Rule;
var NoImportsWithRegexWalker = /** @class */ (function (_super) {
    __extends(NoImportsWithRegexWalker, _super);
    function NoImportsWithRegexWalker() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    NoImportsWithRegexWalker.prototype.visitImportDeclaration = function (node) {
        var options = this.getOptions();
        for (var _i = 0, options_1 = options; _i < options_1.length; _i++) {
            var regStr = options_1[_i];
            var importFrom = node.moduleSpecifier.getText();
            var reg = new RegExp(regStr);
            if (importFrom.match(reg)) {
                this.addFailure(this.createFailure(node.moduleSpecifier.getStart(), node.moduleSpecifier.getWidth(), "importing from " + regStr + " is prohibited."));
            }
        }
        _super.prototype.visitImportDeclaration.call(this, node);
    };
    return NoImportsWithRegexWalker;
}(Lint.RuleWalker));
