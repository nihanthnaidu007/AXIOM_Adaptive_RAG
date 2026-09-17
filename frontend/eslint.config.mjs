import js from "@eslint/js";
import globals from "globals";
import react from "eslint-plugin-react";
import reactHooks from "eslint-plugin-react-hooks";
import jsxA11y from "eslint-plugin-jsx-a11y";

export default [
  {
    ignores: [
      "build/**",
      "node_modules/**",
      "craco.config.js",
      "postcss.config.js",
      "tailwind.config.js",
    ],
  },
  js.configs.recommended,
  {
    files: ["src/**/*.{js,jsx}"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
      globals: {
        ...globals.browser,
        // CRA's webpack config injects process.env at build time.
        process: "readonly",
      },
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
    settings: { react: { version: "detect" } },
    plugins: {
      react,
      "react-hooks": reactHooks,
      "jsx-a11y": jsxA11y,
    },
    rules: {
      ...(react.configs.flat?.recommended?.rules ?? {}),
      // CRA compiles JSX with the automatic runtime — React import not required.
      "react/react-in-jsx-scope": "off",
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
      "no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
      "react/prop-types": "off",
    },
  },
  // Vendored shadcn/ui primitives — kept diffable against upstream; the
  // cmdk-input-wrapper marker attribute is upstream convention with no local
  // CSS references. Placed last: flat config's final matching block wins.
  {
    files: ["src/components/ui/**"],
    rules: { "react/no-unknown-property": "off" },
  },
];
