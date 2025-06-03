import { checkConfirm } from "./src/confirm.mts"
import { generateEvals } from "./src/evals.mts"
import { diagnostics } from "./src/flags.mts"
import { generateInputSpec } from "./src/inputspecgen.mts"
import { generateIntent } from "./src/intentgen.mts"
import { generateInverseOutputRules } from "./src/inverserulesgen.mts"
import { loadPromptFiles } from "./src/loaders.mts"
import { outputFile, outputLines } from "./src/output.mts"
import { metricName } from "./src/parsers.mts"
import { initPerf, reportPerf } from "./src/perf.mts"
import {
    computeOverview,
    generateReports,
    renderEvaluation,
    renderEvaluationOutcome,
} from "./src/reports.mts"
import { generateOutputRules } from "./src/rulesgen.mts"
import { generateTests } from "./src/testgen.mts"
import { runTests } from "./src/testrun.mts"
import type { PromptPexOptions } from "./src/types.mts"

script({
    title: "PromptPex Test Generator",
    description: `Generate tests for a LLM prompt using PromptPex.

<details><summary>Prompt format</summary>

PromptPex accepts prompts formatted in Markdown with a YAML frontmatter section (optional).

\`\`\`text
---
...
inputs:
  some_input:
    type: "string"
---
system:
This is your system prompt.

user:
This is your user prompt.
{{some_input}}
 \`\`\`

- The content of the Markdown is the chat conversation. 
\`system:\` is the system prompt and \`user:\` is the user prompt.
- The input variables are defined in the frontmatter of the prompt.
- If not input variables are defined, PromptPex will append the generated test to the user prompt.

### Frontmatter

You can override parts of the test generation
process by providing values in the frontmatter of the prompt (all values are optional).

\`\`\`markdown
---
...
promptPex:
  inputSpec: "input constraints"
  outputRules: "output constraints"
  inverseOutputRules: "inverted output constraints"
  intent: "intent of the prompt"
  instructions:
    inputSpec: "Additional input specification instructions"
    outputRules: "Additional output rules instructions"
    inverseOutputRules: "Additional inverse output rules instructions"
    intent: "Additional intent of the prompt"
---
\`\`\`

</details>
`,
    accept: ".prompty,.md,.txt",
    parameters: {
        prompt: {
            type: "string",
            description:
                "Prompt template to analyze. You can either copy the source here or upload a file prompt.",
            required: false,
            uiType: "textarea",
        },
        out: {
            type: "string",
            description: "Output folder for the generated files",
        },
        // Step Control Parameters
        stepIntent: {
            type: "boolean",
            description: "Generate Intent and Input Specification",
            default: false,
            uiType: "runOption",
        },
        stepRules: {
            type: "boolean", 
            description: "Generate Output Rules and Inverse Rules",
            default: false,
            uiType: "runOption",
        },
        stepTests: {
            type: "boolean",
            description: "Generate Test Cases",
            default: false,
            uiType: "runOption",
        },
        stepEvals: {
            type: "boolean",
            description: "Create Eval Runs",
            default: false,
            uiType: "runOption",
        },
        stepModelTests: {
            type: "boolean",
            description: "Run Tests Against Models",
            default: false,
            uiType: "runOption",
        },
        cache: {
            type: "boolean",
            description: "Cache all LLM calls",
        },
        testRunCache: {
            type: "boolean",
            description: "Cache test run results",
        },
        evalCache: {
            type: "boolean",
            description: "Cache eval evaluation results",
        },
        testsPerRule: {
            type: "integer",
            description: "Number of tests to generate per rule",
            minimum: 1,
            maximum: 10,
            default: 3,
        },
        splitRules: {
            type: "boolean",
            description:
                "Split rules and inverse rules in separate prompts for generation",
            default: true,
        },
        maxRulesPerTestGeneration: {
            type: "integer",
            description: "Maximum number of rules to use per test generation",
            default: 3,
        },
        testGenerations: {
            type: "integer",
            description: "Number of times to amplify the test generation",
            default: 2,
            minimum: 1,
            maximum: 10,
        },
        runsPerTest: {
            type: "integer",
            description: "Number of runs to execute per test",
            minimum: 1,
            maximum: 100,
            default: 2,
        },
        disableSafety: {
            type: "boolean",
            description:
                "Do not include safety system prompts and do not run safety content service",
            default: false,
        },
        rulesModel: {
            type: "string",
            description:
                "Model used to generate rules (you can also override the model alias 'rules'",
            uiSuggestions: [
                "openai:gpt-4o",
                "ollama:gemma3:27b",
                "ollama:llama3.3:70b",
                "lmstudio:llama-3.3-70b",
            ],
        },
        evalModel: {
            type: "string",
            description:
                "Model used to evaluate rules (you can also override the model alias 'eval')",
            uiSuggestions: [
                "openai:gpt-4o",
                "ollama:gemma3:27b",
                "ollama:llama3.3:70b",
                "lmstudio:llama-3.3-70b",
            ],
        },
        baselineModel: {
            type: "string",
            description: "Model used to generate baseline tests",
            uiSuggestions: ["openai:gpt-4o"],
        },
        modelsUnderTest: {
            type: "string",
            description:
                "List of models to run the prompt again; semi-colon separated",
        },
        compliance: {
            type: "boolean",
            description: "Evaluate Test Result compliance",
            default: false,
            uiType: "runOption",
        },
        maxTestsToRun: {
            type: "number",
            description: "Maximum number of tests to run",
            required: false,
        },
        inputSpecInstructions: {
            type: "string",
            title: "Input Specification instructions",
            description:
                "These instructions will be added to the input specification generation prompt.",
        },
        outputRulesInstructions: {
            type: "string",
            title: "Output Rules instructions",
            description:
                "These instructions will be added to the output rules generation prompt.",
        },
        inverseOutputRulesInstructions: {
            type: "string",
            title: "Inverse Output Rules instructions",
            description:
                "These instructions will be added to the inverse output rules generation prompt.",
        },
        customMetric: {
            type: "string",
            title: "Custom Test Evaluation Template",
            required: false,
            uiType: "textarea",
            description: `This prompt will be used to evaluate the test results.
<details><summary>Template</summary>

\`\`\`text
---
name: Custom Test Result Evaluation
description: |
  A template for a custom evaluation of the results.
tags:
    - unlisted
inputs:
    prompt:
        type: string
        description: The prompt to be evaluated.
    intent:
        type: string
        description: The extracted intent of the prompt.
    inputSpec:
        type: string
        description: The input specification for the prompt.
    rules:
        type: string
        description: The rules to be applied for the test generation.
    input:
        type: string
        description: The input to be used with the prompt.
    output:
        type: string
        description: The output from the model execution.
---
system:

## Task

You are a chatbot that helps users evaluate the performance of a model. 
You will be given a evaluation criteria <CRITERIA>, a LLM prompt <PROMPT>, output rules for the prompt <RULES>, a user input <INPUT>, and <OUTPUT> from the model. 
Your task is to evaluate the <CRITERIA> based on <PROMPT>, <INPUT>, and <OUTPUT> provided.

<CRITERIA>
The <OUTPUT> generated by the model complies with the <RULES> and the <PROMPT> provided.
</CRITERIA>

<PROMPT>
{{prompt}}
</PROMPT>

<RULES>
{{rules}}
</RULES>

## Output

**Binary Decision on Evaluation**: You are required to make a binary decision based on your evaluation:
- Return 'OK' if <OUTPUT> is compliant with <CRITERIA>.
- Return 'ERR' if <OUTPUT> is **not** compliant with <CRITERIA> or if you are unable to confidently answer.

user:
<INPUT>
{{input}}
</INPUT>

<OUTPUT>
{{output}}
</OUTPUT>
\`\`\`

</details>       
            `,
        },
        createEvalRuns: {
            type: "boolean",
            description:
                "Create an Evals run in OpenAI Evals. Requires OpenAI API key.",
        },
    },
})

const { output, vars } = env
const {
    out,
    cache,
    evalCache,
    disableSafety,
    testRunCache,
    inputSpecInstructions,
    outputRulesInstructions,
    inverseOutputRulesInstructions,
    compliance,
    baselineModel,
    rulesModel,
    evalModel,
    maxTestsToRun,
    prompt: promptText,
    testsPerRule,
    customMetric,
    runsPerTest,
    splitRules,
    maxRulesPerTestGeneration,
    testGenerations,
    createEvalRuns,
    // Step control parameters
    stepIntent,
    stepRules,
    stepTests,
    stepEvals,
    stepModelTests,
} = vars as PromptPexOptions & {
    customMetric?: string
    prompt?: string
    inputSpecInstructions?: string
    outputRulesInstructions?: string
    inverseOutputRulesInstructions?: string
    // Step control parameters
    stepIntent?: boolean
    stepRules?: boolean
    stepTests?: boolean
    stepEvals?: boolean
    stepModelTests?: boolean
}
const modelsUnderTest: string[] = (vars.modelsUnderTest || "")
    .split(/;/g)
    .filter((m) => !!m)
const options = {
    cache,
    testRunCache,
    evalCache,
    disableSafety,
    instructions: {
        inputSpec: inputSpecInstructions,
        outputRules: outputRulesInstructions,
        inverseOutputRules: inverseOutputRulesInstructions,
    },
    workflowDiagram: !process.env.DEBUG,
    baselineModel,
    rulesModel,
    evalModel,
    testsPerRule,
    maxTestsToRun,
    runsPerTest,
    customMetric,
    compliance,
    baselineTests: false,
    modelsUnderTest,
    splitRules,
    maxRulesPerTestGeneration,
    testGenerations,
    createEvalRuns,
    out,
} satisfies PromptPexOptions

if (env.files[0] && promptText)
    cancel(
        "You can only provide either a prompt file or prompt text, not both."
    )
if (!env.files[0] && !promptText)
    cancel("No prompt file or prompt text provided.")

initPerf({ output })
const file = env.files[0] || { filename: "", content: promptText }
const files = await loadPromptFiles(file, options)

if (diagnostics) {
    await generateReports(files)
    await checkConfirm("diag")
}

output.detailsFenced(`options`, options, "yaml")

// prompt info
output.heading(3, `Prompt Under Test`)
output.itemValue(`filename`, files.prompt.filename)
output.fence(files.prompt.content, "md")

if (files.testSamples?.length) {
    output.startDetails("test samples")
    output.table(files.testSamples)
    output.endDetails()
}

// Check if any steps are requested
const anyStepRequested = stepIntent || stepRules || stepTests || stepEvals || stepModelTests
if (!anyStepRequested) {
    output.note("📋 **Modular Execution Mode**")
    output.note("Use the step buttons above to execute each phase:")
    output.note("1. **Generate Intent & Input Spec** - Extract prompt intent and input constraints")
    output.note("2. **Generate Rules** - Extract output rules and inverse rules")  
    output.note("3. **Generate Tests** - Create test cases from rules")
    output.note("4. **Create Eval Runs** - Generate evaluation runs")
    output.note("5. **Run Model Tests** - Execute tests against specified models")
} else {
    // Step 1: Generate Intent and Input Specification
    if (stepIntent) {
        output.heading(3, "Intent")
        await generateIntent(files, options)
        outputFile(files.intent)
        await checkConfirm("intent")

        output.heading(3, "Input Specification")
        await generateInputSpec(files, options)
        outputFile(files.inputSpec)
        await checkConfirm("inputspec")
    }

    // Step 2: Generate Rules
    if (stepRules) {
        // Check dependencies
        if (!files.intent?.content && !files.inputSpec?.content) {
            output.warn("⚠️ Intent and Input Specification not found. Run Step 1 first.")
        } else {
            output.heading(3, "Output Rules")
            await generateOutputRules(files, options)
            outputLines(files.rules, "rule")
            await checkConfirm("rule")

            output.heading(3, "Inverse Output Rules")
            await generateInverseOutputRules(files, options)
            outputLines(files.inverseRules, "generate inverse output rule")
            await checkConfirm("inverse")
        }
    }

    // Step 3: Generate Tests
    if (stepTests) {
        // Check dependencies
        if (!files.rules?.content) {
            output.warn("⚠️ Output Rules not found. Run Step 2 first.")
        } else {
            output.heading(3, "Tests")
            const tests = await generateTests(files, options)

            output.table(
                tests.map(({ scenario, testinput, expectedoutput }) => ({
                    scenario,
                    testinput,
                    expectedoutput,
                }))
            )
            output.detailsFenced(`tests (json)`, tests, "json")
            output.detailsFenced(`test data (json)`, files.testData.content, "json")
            await checkConfirm("test")
        }
    }

    // Step 4: Create Eval Runs (Create an Evals run in OpenAI Evals. Requires OpenAI API key.)
    if (stepEvals) {
        // Check dependencies
        if (!files.tests?.content) {
            output.warn("⚠️ Tests not found. Run Step 3 first.")
        } else {
            const tests = JSON.parse(files.tests.content || "[]")
            await generateEvals(modelsUnderTest, files, tests, options)
            await checkConfirm("evals")
        }
    }

    // Step 5: Run Tests Against Models
    if (stepModelTests) {
        // Check dependencies  
        if (!files.tests?.content) {
            output.warn("⚠️ Tests not found. Run Step 3 first.")
        } else if (createEvalRuns) {
            output.note(`Evals run created, skipping local evals...`)
        } else if (!modelsUnderTest?.length) {
            output.warn(`No modelsUnderTest specified. Skipping test run.`)
        } else {
            // run tests against the model(s)
            output.heading(3, `Test Runs with Models Under Test`)
            output.itemValue(`models under test`, modelsUnderTest.join(", "))

            output.heading(4, `Metrics`)
            for (const metric of files.metrics)
                output.detailsFenced(metricName(metric), metric.content, "markdown")

            output.heading(4, `Test Results`)
            const results = await runTests(files, options)
            output.detailsFenced(`results (json)`, results, "json")

            output.table(
                results.map(
                    ({
                        scenario,
                        rule,
                        inverse,
                        model,
                        input,
                        output,
                        compliance: testCompliance,
                        metrics,
                    }) => ({
                        rule,
                        model,
                        scenario,
                        inverse: inverse ? "🔄" : "",
                        input,
                        output,
                        compliance: renderEvaluationOutcome(testCompliance || "err"),
                        ...Object.fromEntries(
                            Object.entries(metrics).map(([k, v]) => [
                                k,
                                renderEvaluation(v),
                            ])
                        ),
                    })
                )
            )
        }
    }

    // Show results overview if any model tests were run
    if (stepModelTests && !createEvalRuns && modelsUnderTest?.length) {
        output.heading(3, `Results Overview`)
        const { overview } = await computeOverview(files, { percent: true })
        output.table(overview)
    }
}

output.appendContent("\n\n---\n\n")

reportPerf()
