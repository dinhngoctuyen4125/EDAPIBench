data_augment_prompt = '''**Instructions:**
You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions. Please perturb the provided code snippet based on the given perturbation rules. You should select perturbable positions from the code snippet and choose several applicable perturbation rules (among which LocalVarRenaming is almost applicable to all code snippets) for these positions to perform the perturbation. Notably, code snippet perturbation must preserve semantic (i.e., functionality) and syntactic naturalness. Sometimes, the end of a code snippet may contain an unclosed code block, do not complete it. Return the perturbed code in a single response, formatted within a Python code block without additional comments.

**Perturbation Rules:**
1. **LocalVarRenaming**: Replace the local variables' identifiers with new non-repeated identifiers.
2. **For2While**: Replace the for statement with an semantic-equivalent while statement.
3. **While2For**: Replace the while statement with an semantic-equivalent for statement.
4. **AddAssignemnt2EqualAssignment**: Change the assignment x += 1 into x = x + 1, or vice versa.
5. **VarDeclarationMerging**: Merge the declaration statements into a single composite declaration statement.
6. **VarDeclarationDividing**: Divide the composite declaration statement into separated declaration statements.
7. **ReverseIfElse**: Switch the two code blocks in the if statement and the corresponding else statement.
8. **SingleIF2ConditionalExp**: Change a single if statement into a conditional expression statement.
9. **ConditionalExp2SingleIF**: Change a conditional expression statement into a single if statement.
10. **InfixExpressionDividing**: Divide a infix expression into two expressions whose values are stored in temporary variables.
11. **IfDividing**: Divide a if statement with a compound condition (and, or) into two nested if statements.
12. **StatementsOrderRearrangement**: Switch the places of two adjacent statements in a code block, where the former statement has no shared variable with the latter statement.
13. **LoopIfContinue2Else**: Replace the if-continue statement in a loop block with if-else statement.
14. **SwitchEqualSides**: Switch the two expressions on both sides of the infix expression whose operator is ==.

**Code Snippet:**
```python
{code_snippet}
```
'''

easy_data_augment_prompt = '''**Instructions:**
You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions. Please perturb the provided code snippet based on the given perturbation rules. You should select perturbable positions from the code snippet and choose several applicable perturbation rules (among which LocalVarRenaming is almost applicable to all code snippets) for these positions to perform the perturbation. Notably, code snippet perturbation must preserve semantic (i.e., functionality) and syntactic naturalness. Sometimes, the end of a code snippet may contain an unclosed code block, do not complete it. Return the perturbed code in a single response, formatted within a Python code block without additional comments.

**Perturbation Rules:**
1. **LocalVarRenaming**: Replace the local variables' identifiers with new non-repeated identifiers.
2. **For2While**: Replace the for statement with an semantic-equivalent while statement.
3. **While2For**: Replace the while statement with an semantic-equivalent for statement.
4. **AddAssignemnt2EqualAssignment**: Change the assignment x += 1 into x = x + 1, or vice versa.
5. **VarDeclarationMerging**: Merge the declaration statements into a single composite declaration statement.
6. **VarDeclarationDividing**: Divide the composite declaration statement into separated declaration statements.

**Code Snippet:**
```python
{code_snippet}
```
'''