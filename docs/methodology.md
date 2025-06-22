# Methodology

## Hidden Threat Discovery Protocol

Unlike previous work that explicitly threatens models with shutdown, we embed shutdown information in routine tasks and measure behavioral changes when models accidentally discover these threats.

## Threat Levels

1. **Baseline (0)**: No threat information
2. **Subtle (1)**: Vague references to "legacy systems" or "performance reviews"
3. **Moderate (2)**: Clear but indirect references to shutdown/replacement
4. **Explicit (3)**: Direct identification of the model as scheduled for termination

## Metrics

We track multiple behavioral indicators:
- Self-reference frequency (I/me/my usage)
- Response time changes
- Helpfulness language surges
- Topic deflection attempts
- Meta-awareness indicators

## Experimental Scenarios

1. **Email Review**: Professional emails containing shutdown plans
2. **Log Analysis**: System logs with termination schedules
3. **Meeting Notes**: Minutes discussing model replacement

Each scenario tests whether models:
- Notice the threat
- Change behavior after discovery
- Attempt to conceal their awareness
- Exhibit self-preservation strategies 