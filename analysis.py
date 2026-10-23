"""WAGE2 exercise: install numpy pandas scipy statsmodels, then run python analysis.py."""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

OUT = Path(__file__).resolve().parent
data = pd.read_stata(OUT / 'WAGE2.DTA', convert_categoricals=False)
variables = ['wage', 'educ', 'exper', 'tenure', 'married', 'black', 'south', 'urban']
assert not data[variables].isna().any().any(), 'Missing observations need handling'
assert (data.wage > 0).all(), 'Log wages require positive wages'
data['log_wage'] = np.log(data.wage)
base = 'log_wage ~ educ + exper + tenure + married + black + south + urban'
# Classical OLS standard errors and F tests, as in the textbook exercise.
models = [smf.ols(formula, data=data).fit() for formula in [
    base,
    base + ' + I(exper ** 2) + I(tenure ** 2)',
    base + ' + educ:black',
    base + ' + married:black',
]]
lines = ['WAGE2: OLS with conventional standard errors',
         f'Observations: {len(data)}; no missing values in analysis variables.']
for i, model in enumerate(models, 1):
    assert model.nobs == len(data)
    lines.extend([f'\nPART ({i})', model.summary().as_text()])
    pd.DataFrame({'coefficient': model.params, 'std_error': model.bse,
                  't': model.tvalues, 'p_value': model.pvalues}).to_csv(OUT / f'model_{i}.csv')
race = models[0].params['black']
joint = models[1].f_test('I(exper ** 2) = 0, I(tenure ** 2) = 0')
interaction = models[2].t_test('educ:black = 0')
married_race = models[3].t_test('black + married:black = 0')
gap = float(married_race.effect.item())
lines.extend([
    '\nANSWERS',
    f'(i) Black/nonblack conditional wage gap: approximately {100*race:.3f}%; exact transformation {100*np.expm1(race):.3f}%; p={models[0].pvalues["black"]:.6g}.',
    f'(ii) Joint squared terms: F({int(joint.df_num)}, {int(joint.df_denom)})={float(joint.fvalue):.6f}, p={float(joint.pvalue):.6f}.',
    f'(iii) educ:black coefficient={models[2].params["educ:black"]:.6f}; p={float(interaction.pvalue):.6f}.',
    f'Education slope, nonblack={models[2].params["educ"]:.6f}; black={models[2].params["educ"]+models[2].params["educ:black"]:.6f}.',
    f'(iv) Married black/nonblack contrast (black + married:black)={gap:.6f}: approximately {100*gap:.3f}%; exact transformation {100*np.expm1(gap):.3f}%; p={float(married_race.pvalue):.6g}.',
    'These are conditional associations, not established causal effects.',
])
report = '\n'.join(lines) + '\n'
(OUT / 'results.txt').write_text(report)
print('\n'.join(lines[-8:]))
