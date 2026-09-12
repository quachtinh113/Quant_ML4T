# calendars/ - Trading Calendar Helpers

Calendar utilities for session-aware workflows.

## Files

| File | Purpose |
|------|---------|
| `base.py` | common calendar interfaces |
| `equity.py` | equity-market calendar behavior |
| `crypto.py` | always-open crypto calendar behavior |

## Notes

- these helpers matter when labeling or horizon logic depends on trading sessions
- public calendar-facing APIs are surfaced mainly through the labeling module
