"""Summarize a frozen radial EMS series without accepting unfinished evolutions."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np
from scipy.interpolate import CubicSpline


def history(path):
    text=path.read_text()
    if not text.endswith('\n'):text=text[:text.rfind('\n')+1]
    rows=list(csv.DictReader(text.splitlines()))
    values={k:np.array([float(r[k]) for r in rows]) for k in rows[0]}
    assert all(np.isfinite(v).all() for v in values.values()),path
    assert np.all(np.diff(values['time'])>0),path
    return values


def tail(values,key,start,end):
    t,y=values['time'],values[key]
    spline=CubicSpline(t,y)
    x=np.r_[start,t[(t>start)&(t<end)],end]
    z=spline(x)
    mean=float(spline.integrate(start,end)/(end-start))
    slope=float(np.polyfit(x-start,z,1)[0])
    return dict(mean=mean,sampled_peak_to_peak=float(np.ptp(z)),linear_slope=slope,
                relative_range=float(np.ptp(z)/abs(mean)),
                relative_linear_drift=float(abs(slope)*(end-start)/abs(mean)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--series',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--reference-json',type=Path)
    parser.add_argument('--tail-duration',type=float,default=20)
    parser.add_argument('--stationarity-relative-tolerance',type=float,default=.001)
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    manifest=json.loads(args.series.read_text());binary=Path(manifest['binary'])
    assert hashlib.sha256(binary.read_bytes()).hexdigest()==manifest['binary_sha256']
    assert args.tail_duration>0 and args.stationarity_relative_tolerance>0
    target=float(manifest['target']['initial_bounded_time_Q_units'])
    data={}
    for folder in sorted(args.series.parent.glob('n*')):
        if folder.name[1:].isdigit() and (folder/'ems_spherical_history.csv').is_file():
            data[int(folder.name[1:])]=history(folder/'ems_spherical_history.csv')
    assert len(data)>=2
    args.output.mkdir(parents=True,exist_ok=True)
    complete=all(v['time'][-1]>=target-1.e-10 for v in data.values()) and len(data)==3
    end=min(target,min(v['time'][-1] for v in data.values()))
    start=max(0,end-args.tail_duration)
    report=dict(status='bounded_series_complete' if complete else 'awaiting_common_target',
        target_Q_time=target,common_end_Q_time=float(end),tail_window_Q_time=[float(start),float(end)],
        scope='Radial non-PG RN scalarization branch only; this is not the symmetry-broken-background or axion model.',
        binary_sha256=manifest['binary_sha256'],source_sha256=manifest['source_sha256'],
        constraint_region='Native history: 1.1 times instantaneous AH coordinate radius < r <50; r^2 dr weighting; all interior cells excluded.',
        temporal_interpolation='Cubic spline on native diagnostic samples; sampled ranges and linear trends are diagnostic estimates.',
        stationarity_relative_tolerance=args.stationarity_relative_tolerance,
        tolerance_scope='Explicit pilot diagnostic tolerance, not a threshold specified by the paper.',cases={})
    reference=None
    if args.reference_json:
        reference=json.loads(args.reference_json.read_text());report['reference']=reference
        report['reference_scope']='High-resolution plateau estimated from published PDF; computational grids differ. This is not an author-supplied data table.'
    for n,v in data.items():
        phi=tail(v,'abs_phi_h',start,end);mass=tail(v,'irreducible_mass',start,end)
        row=dict(last_Q_time=float(v['time'][-1]),common_time_Q=end,
                 phi_h_at_common_time=float(CubicSpline(v['time'],v['abs_phi_h'])(end)),
                 Mirr_at_common_time=float(CubicSpline(v['time'],v['irreducible_mass'])(end)),
                 H_RMS_at_common_time=float(CubicSpline(v['time'],v['H_exterior_RMS'])(end)),
                 M_RMS_at_common_time=float(CubicSpline(v['time'],v['M_exterior_RMS'])(end)),
                 phi_tail=phi,Mirr_tail=mass,
                 exterior_chi_floor_cells_max=int(v['chi_floor_exterior_cells'].max()),
                 exterior_lapse_floor_cells_max=int(v['lapse_floor_exterior_cells'].max()))
        row['tail_stationary_within_diagnostic_tolerance']=bool(complete and
            max(phi['relative_range'],phi['relative_linear_drift'],mass['relative_range'])<args.stationarity_relative_tolerance)
        if reference:
            ref=reference['late_value_estimate']
            row['phi_endpoint_relative_difference_from_PDF_plateau']=abs(row['phi_h_at_common_time']/ref-1)
            row['phi_tail_mean_relative_difference_from_PDF_plateau']=abs(phi['mean']/ref-1)
        report['cases'][str(n)]=row
    if complete:
        ns=np.array(sorted(data));norms=np.array([report['cases'][str(n)]['H_RMS_at_common_time'] for n in ns])
        report['H_RMS_resolution_log_slope']=float(-np.polyfit(np.log(ns),np.log(norms),1)[0])
        report['H_RMS_decreases_with_resolution']=bool(np.all(np.diff(norms)<0))
        a,b=[report['cases'][str(n)] for n in ns[-2:]]
        report['two_finest_phi_endpoint_relative_difference']=abs(a['phi_h_at_common_time']/b['phi_h_at_common_time']-1)
        report['two_finest_phi_tail_mean_relative_difference']=abs(a['phi_tail']['mean']/b['phi_tail']['mean']-1)
        report['scientific_status']='Review endpoint agreement, tail stationarity and exterior constraint convergence together; no automatic full-paper acceptance.'
    (args.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    with (args.output/'SUMMARY.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['radial_points','common_Q_time','phi_h','Mirr','H_exterior_RMS','phi_tail_mean','phi_tail_relative_range','tail_stationary'])
        for n,r in report['cases'].items():
            writer.writerow([n,end,r['phi_h_at_common_time'],r['Mirr_at_common_time'],r['H_RMS_at_common_time'],r['phi_tail']['mean'],r['phi_tail']['relative_range'],r['tail_stationary_within_diagnostic_tolerance']])
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
        for n,v in sorted(data.items()):
            label=f'N_R={n}'
            ax[0,0].plot(v['time'],v['abs_phi_h'],label=label)
            ax[0,1].plot(v['time'],v['abs_phi_h'],label=label)
            ax[1,0].semilogy(v['time'],v['H_exterior_RMS'],label=label)
            ax[1,1].plot(v['time'],v['irreducible_mass'],label=label)
        ax[0,1].set_xlim(max(0,end-60),end)
        ax[0,1].set_ylim(.1412,.1428)
        if reference:
            ax[0,1].axhline(reference['late_value_estimate'],color='black',ls='--',lw=1,label='PDF Fig.6 plateau estimate')
        for a,title,ylabel in zip(ax.flat,['Scalarization','Late horizon scalar','Exterior constraints','Horizon mass'],['|Phi_h|','|Phi_h|','H RMS','M_irr']):
            a.set(title=title,xlabel='t / Q (Q=1)',ylabel=ylabel);a.grid(alpha=.2);a.legend(fontsize=8)
        fig.suptitle('Spherical non-PG BSSN: M0=2, Q=1, coupling alpha=160')
        fig.savefig(args.output/'radial-comparison.png',dpi=180)
        fig.savefig(args.output/'radial-comparison.pdf')
        plt.close(fig)
    print(json.dumps(dict(status=report['status'],common_end_Q_time=end,cases=report['cases']),indent=2))


if __name__=='__main__':main()
