"""Post-reveal accounting of the frozen Fine64 versus full-query alarms."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    paths = [HERE / name for name in ('FORECAST.json', 'SUMMARY.json')]
    forecast, summary = [json.loads(path.read_text()) for path in paths]
    rows = []
    for prediction, scored in zip(forecast['panels'], summary['panels'], strict=True):
        assert prediction['index'] == scored['index']
        ids = prediction['candidates']
        truth = dict(zip(ids, scored['truth_gap'], strict=True))
        sparse = dict(zip(ids, prediction['gaps']['fine_64'], strict=True))
        full = dict(zip(ids, prediction['gaps']['fine_all'], strict=True))
        lost = set(scored['hit_ids']['fine_all']) - set(scored['hit_ids']['fine_64'])
        added = set(scored['hit_ids']['fine_64']) - set(scored['hit_ids']['fine_all'])
        difference = scored['hit']['fine_64'] - scored['hit']['fine_all']
        assert difference == len(added) - len(lost)
        rows.append(dict(
            panel=prediction['index'], hit_difference=difference,
            alarms_identical=set(prediction['flags']['fine_64']) == set(prediction['flags']['fine_all']),
            lost_hits=[dict(id=i, true_gap=truth[i], gap64=sparse[i], gap_all=full[i]) for i in sorted(lost)],
            added_hits=sorted(added),
            fine64_edge_rms=prediction['reference']['graphs']['fine']['64']['relative_edge_rms'],
            raw64_edge_rms=prediction['reference']['graphs']['raw']['64']['relative_edge_rms'],
            fine64_scale_max_error=max(abs(v) for v in scored['scale_relative_error']['fine']['64'])))
    output = dict(
        role='post_reveal_description_only',
        sources={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        independent_panels=len(rows),
        hit_difference=sum(row['hit_difference'] for row in rows),
        panels_with_fewer_hits=sum(row['hit_difference'] < 0 for row in rows),
        panels_with_same_hit_count=sum(row['hit_difference'] == 0 for row in rows),
        identical_alarm_panels=sum(row['alarms_identical'] for row in rows),
        edge_rms_ranges={rep: [min(row[f'{rep}64_edge_rms'] for row in rows),
                               max(row[f'{rep}64_edge_rms'] for row in rows)] for rep in ('fine', 'raw')},
        panels=rows)
    assert output['hit_difference'] == summary['primary']['fine64_vs_fine_all']['delta_hits']
    with (HERE / 'RETENTION_DIAGNOSTIC.json').open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
