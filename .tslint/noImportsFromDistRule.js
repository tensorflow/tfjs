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
        return this.applyWithWalker(new NoImportsFromDistWalker(sourceFile, this.getOptions()));
    };
    Rule.FAILURE_STRING = "importing from dist/ is prohibited. Please use public API";
    return Rule;
}(Lint.Rules.AbstractRule));
exports.Rule = Rule;
var NoImportsFromDistWalker = /** @class */ (function (_super) {
    __extends(NoImportsFromDistWalker, _super);
    function NoImportsFromDistWalker() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    NoImportsFromDistWalker.prototype.visitImportDeclaration = function (node) {
        var importFrom = node.moduleSpecifier.getText();
        var reg = /@tensorflow\/tfjs[-a-z]*\/dist/;
        if (importFrom.match(reg)) {
            var fix = new Lint.Replacement(node.moduleSpecifier.getStart(), node.moduleSpecifier.getWidth(), importFrom.replace(/\/dist[\/]*/, ''));
            this.addFailure(this.createFailure(node.moduleSpecifier.getStart(), node.moduleSpecifier.getWidth(), Rule.FAILURE_STRING, fix));
        }
        _super.prototype.visitImportDeclaration.call(this, node);
    };
    return NoImportsFromDistWalker;
}(Lint.RuleWalker));
