export type PromptPexModelAliases = OptionsOrString<
    "rules" | "eval" | "large" | "baseline"
>

export interface PromptPexPrompts {
    /**
     * Input specifications, overrides input spec generation
     */
    inputSpec?: string
    /**
     * Output rules, overrides output rules generation
     */
    outputRules?: string
    /**
     * Inverse output rules, overrides inverse output rules generation
     */
    inverseOutputRules?: string
    /**
     * Prompt intent, overrides intent generation
     */
    intent?: string
}

export interface PromptPexOptions extends PromptPexLoaderOptions {
    /**
     * Generate temperature for requests
     */
    temperature?: number

    /**
     * Add PromptPex prompts to the output
     */
    outputPrompts?: boolean

    /**
     * Emit diagrams in output
     */
    workflowDiagram?: boolean

    /**
     * Additional instructions
     */
    instructions?: PromptPexPrompts
    /**
     * Custom model aliases
     */
    modelAliases?: Partial<Record<PromptPexModelAliases, string>>

    /**
     * Caches results in the file system
     */
    evalCache?: boolean

    /**
     * Cache runTest results
     */
    testRunCache?: boolean

    /**
     * Model used to generate rules
     */
    rulesModel?: string

    /**
     * Model used to evaluate rules
     */
    evalModel?: ModelType

    /**
     * Model used to generate baseline tests
     */
    baselineModel?: ModelType

    /**
     * Number of tests to generate per rule
     */
    testsPerRule?: number

    /**
     * Number of run to execute per test
     */
    runsPerTest?: number

    /**
     * Evaluate test result coverage and validity
     */
    compliance?: boolean

    /**
     * Generate and evaluate baseline tests
     */
    baselineTests?: boolean

    /**
     * Maximum number of tests to run
     */
    maxTestsToRun?: number

    /**
     * Maximum number of output rules (and inverse rules) to use
     */
    maxRules?: number

    /**
     * Cache applied to all prompts, expect run test.
     */
    cache?: boolean | string

    /**
     * List of models to run the prompt against
     */
    modelsUnderTest?: ModelType[]

    /**
     * Split rules/inverse rules in separate prompts
     */
    splitRules?: boolean

    /**
     * Maximum number of rules to use per test generation
     */
    maxRulesPerTestGeneration?: number

    /**
     * Number of times to amplify the test generation, default is 1
     */
    testGenerations?: number

    /**
     * Creates a new eval run in OpenAI. Requires OpenAI API key.
     */
    createEvalRuns?: boolean

    /**
     * Mutate one OR into IR if mutateRule is true.
     * If false, the inverse rules will be generated as is.
     */
    mutateRule?: boolean
    
    /**
     * Compliance threshold for iteration system (0-1, default 0.5)
     */
    complianceThreshold?: number
    
    /**
     * Maximum iterations per branch in iteration system (default 5)
     */
    maxIterationsPerBranch?: number
    
    /**
     * Enable the multi-iteration mutation system
     */
    enableMutationSystem?: boolean
}

/**
 * In memory cache of various files involved with promptpex test generation.
 *
 * - Model Used by PromptPex (MPP) - gpt-4o
 * - Model Under Test (MUT) - Model which we are testing against with specific temperature, etc example: gpt-4o-mini
 */
export interface PromptPexContext {
    /** Should write results to files */
    writeResults?: boolean
    /**
     * Prompt folder location if any
     */
    dir?: string
    /**
     * Prompt name
     */
    name: string
    /**
     * Prompt parsed frontmatter section
     */
    frontmatter: PromptPexPromptyFrontmatter
    /**
     * Inputs extracted from the prompt frontmatter
     */
    inputs: Record<string, JSONSchemaSimpleType>
    /**
     * Prompt Under Test
     */
    prompt: WorkspaceFile
    /**
 0  * Prompt Under Test Intent (PUTI)
   */
    intent: WorkspaceFile
    /**
     * Output Rules (OR) - Extracted output constraints of PUT using MPP
     */
    rules: WorkspaceFile
    /**
     * Inverse output rules (IOR) - Negated OR rules
     */
    inverseRules?: WorkspaceFile
    /**
     * Input specification (IS): Extracted input constraints of PUT using MPP
     */
    inputSpec: WorkspaceFile

    /**
     * Baseline Tests (BT) - Zero shot test cases generated for PUT with MPP
     */
    baselineTests: WorkspaceFile

    /**
     * PromptPex Tests (PPT) - Test cases generated for PUT with MPP using IS and OR (test)
     */
    tests: WorkspaceFile

    /**
     * PromptPex Test with resolved input parameters
     */
    testData: WorkspaceFile

    /**
     * Test Output (TO) - Result generated for PPT and BT on PUT with each MUT (the template is PUT)
     */
    testOutputs: WorkspaceFile

    /**
     * Coverage and validate test evals
     */
    testEvals: WorkspaceFile

    /**
     * Groundedness
     */
    ruleEvals: WorkspaceFile

    /**
     * Coverage of rules
     */
    ruleCoverages: WorkspaceFile
    /**
     * Baseline tests validity
     */
    baselineTestEvals: WorkspaceFile

    /**
     * Evaluation metrics prompt files
     */
    metrics: WorkspaceFile[]

    /**
     * Existing test data if any
     */
    testSamples?: Record<string, number | string | boolean>[]

    /**
     * Versions of tooling
     */
    versions: {
        promptpex: string
        node: string
    }
}

export interface PromptPexTest {
    /**
     * Index of the generated test for the given rule. undefined for baseline tests
     */
    testid?: number
    /**
     * Generated by the baseline prompt
     */
    baseline?: boolean
    /**
     * Prompt test input text
     */
    testinput: string
    /**
     * Updated test input
     */
    testinputexpanded?: string
    /**
     * Expected output generated by the PromptPex Test generator
     */
    expectedoutput?: string
    /**
     * Explanation of the test generation process
     */
    reasoning?: string

    /**
     * Scenario name
     */
    scenario?: string

    /**
     * Test generation iteration index
     */
    generation?: number
}

export interface PromptPexTestResult {
    id: string
    promptid: string
    rule: string
    scenario: string
    testinput: string
    inverse?: boolean
    baseline?: boolean
    model: string
    input: string
    output: string
    error?: string

    compliance?: PromptPexEvalResultType
    complianceText?: string

    metrics: Record<string, PromptPexEvaluation>
    
    // Branch information from mutation system
    branchName?: string
    iteration?: number
}

export interface PromptPexTestEval {
    id: string
    promptid: string
    model?: string
    rule: string
    inverse?: boolean
    input: string
    coverage?: PromptPexEvalResultType
    coverageEvalText?: string
    coverageText?: string
    coverageUncertainty?: number
    validity?: PromptPexEvalResultType
    validityText?: string
    validityUncertainty?: number
    error?: string
}

export interface PromptPexRule {
    id: string
    rule: string
    inverseRule: string
    inversed?: boolean
}

export type PromptPexEvalResultType = "ok" | "err" | "unknown"

export interface PromptPexRuleEval {
    id: string
    promptid: string
    rule: string
    groundedText?: string
    grounded?: PromptPexEvalResultType
    error?: string
}

export interface PromptPexLoaderOptions {
    out?: string
    disableSafety?: boolean
    customMetric?: string
}

export interface PromptPexTestGenerationScenario {
    name: string
    instructions?: string
    parameters?: Record<string, number | string | boolean>
}

export interface PromptPexPromptyFrontmatter {
    name?: string
    inputs?: PromptParametersSchema
    outputs?: JSONSchemaObject["properties"]
    instructions?: PromptPexPrompts
    scenarios?: PromptPexTestGenerationScenario[]
    /**
     * A list of samples or file containing samples.
     */
    testSamples?: (string | Record<string, number | string | boolean>)[]
}

export interface PromptPexEvaluation {
    content: string
    uncertainty?: number
    perplexity?: number
    outcome?: PromptPexEvalResultType
    score?: number
    violated_rules?: number[]
}

// Multi-iteration mutation system types
export interface PromptPexMutationNode {
    id: string
    branchName: string
    iteration: number
    mutatedRuleId?: string
    compliance?: number
    testsGenerated: number
    timestamp: string
    results?: PromptPexTestResult[]
    isComplete: boolean
}

export interface PromptPexMutationBranch {
    name: string
    mutatedRuleId?: string
    nodes: PromptPexMutationNode[]
    isComplete: boolean
    bestCompliance?: number
    totalIterations: number
}

export interface PromptPexMutationTree {
    rootBranch: PromptPexMutationBranch
    branches: PromptPexMutationBranch[]
    currentBranch: string
    currentIteration: number
    totalRules: number
    complianceThreshold: number
    maxIterationsPerBranch: number
    isComplete: boolean
    startTime: string
    lastUpdateTime: string
}

export interface PromptPexMutationState {
    tree: PromptPexMutationTree
    availableBranches: string[]
    canContinueIteration: boolean
    canMutateRules: boolean
    nextAction: 'continue_iteration' | 'mutate_rules' | 'complete'
}

export interface PromptPexIterationOptions extends PromptPexOptions {
    complianceThreshold?: number
    maxIterationsPerBranch?: number
    enableMutationSystem?: boolean
    currentBranch?: string
    currentIteration?: number
}
