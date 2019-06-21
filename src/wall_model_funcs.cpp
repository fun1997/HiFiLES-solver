#include "wall_model_funcs.h"
#include <numeric>
double *get_wm_disu_ptr(int in_ele_type, int in_ele, int in_upt, int in_field, struct solution *FlowSol)
{
    return FlowSol->mesh_eles(in_ele_type)->get_wm_disu_ptr(in_ele, in_upt, in_field);
}

double calc_wm_upts_dist(int in_ele_type, int in_ele, int in_local_inter, struct solution *FlowSol, int &out_upt)
{
   return  FlowSol->mesh_eles(in_ele_type)->calc_wm_upts_dist(in_ele, in_local_inter, out_upt);
}

void calc_wall_stress(hf_array<double> &in_u_wm, hf_array<double> &in_uw, double in_dist, hf_array<double> &in_norm, hf_array<double> &out_fn)
{
    int n_dims = in_norm.get_dim(0);
    double rho_wm, inte_wm, v_wm_mag, v_wm_rel_mag, v_wm_n, mu_wm, rt_ratio, ke_wm; //wm params
    double rho_w, inte_w, tw_mag, ke_w, qw, vw_tw;                          //wall params
    hf_array<double> vw(n_dims), v_wm(n_dims), tw(n_dims);

    //calculate aux params
    rho_wm = in_u_wm(0); //input density
    rho_w = in_uw(0);    //wall density
    v_wm_n = 0;
    for (int i = 0; i < n_dims; i++)
        v_wm_n += in_u_wm(i + 1) / in_u_wm(0) * in_norm(i); //wall normal input velocity
    v_wm_mag = 0;
    v_wm_rel_mag = 0;
    for (int i = 0; i < n_dims; i++)
    {
        v_wm(i) = in_u_wm(i + 1) / in_u_wm(0) - in_norm(i) * v_wm_n; //wall parallel input velocity
        vw(i) = in_uw(i + 1) / in_uw(0);                             // wall velocity
        v_wm_rel_mag += pow(v_wm(i) - vw(i), 2);
        v_wm_mag += v_wm(i) * v_wm(i);
    }
    v_wm_mag = sqrt(v_wm_mag); //mag of wall parallel input velocity
    v_wm_rel_mag=sqrt(v_wm_rel_mag); //mag of relative wall parallel input velocity

    ke_wm = 0.;
    ke_w = 0;
    for (int i = 0; i < n_dims; i++)
    {
        ke_wm += 0.5 * pow(in_u_wm(i + 1) / in_u_wm(0), 2);
        ke_w += 0.5 * pow(vw(i), 2);
    }
    inte_wm = in_u_wm(n_dims + 1) / rho_wm - ke_wm; //internal energy of input point
    inte_w = in_uw(n_dims + 1) / rho_w - ke_w;      //internal energy on wall

    //viscosity of input point
    rt_ratio = (run_input.gamma - 1.0) * inte_wm / (run_input.rt_inf);
    mu_wm = (run_input.mu_inf) * pow(rt_ratio, 1.5) * (1 + (run_input.c_sth)) / (rt_ratio + (run_input.c_sth));
    mu_wm = mu_wm + run_input.fix_vis * (run_input.mu_inf - mu_wm);

    //evaluate wall stress
    if (run_input.wall_model == 1) //Werner-Wengle
    {
        double ymatch = 11.8;
        double Rey, Rey_c, uplus, utau;

        Rey_c = ymatch * ymatch;
        Rey = rho_wm * v_wm_rel_mag * in_dist / mu_wm;

        if (Rey < Rey_c)
            uplus = sqrt(Rey);
        else
            uplus = pow(8.3, 0.875) * pow(Rey, 0.125);

        utau = v_wm_rel_mag / uplus;
        tw_mag = rho_wm * utau * utau;

        for (int i = 0; i < n_dims; i++)
            tw(i) = tw_mag * v_wm(i) / v_wm_mag; //stress vector parallel to wall

        // Wall heat flux
        if (Rey < Rey_c)
            qw = (inte_w - inte_wm) * run_input.gamma * tw_mag / (run_input.prandtl * v_wm_rel_mag);
        else
            qw = (inte_w - inte_wm) * run_input.gamma * tw_mag / (run_input.prandtl_t * (v_wm_rel_mag + utau * ymatch * (run_input.prandtl / run_input.prandtl_t - 1.0)));
    }
    else
    {
        FatalError("Wall model not implemented!");
    }

    //calculate wall normal flux
#if defined _ACCELERATE_BLAS || defined _MKL_BLAS || defined _STANDARD_BLAS
    vw_tw = cblas_ddot(n_dims, vw.get_ptr_cpu(), 1, tw.get_ptr_cpu(), 1);
#else
    vw_tw = inner_product(vw.get_ptr_cpu(), vw.get_ptr_cpu(n_dims), tw.get_ptr_cpu(), 0.);
#endif
    out_fn(0) = 0;
    for (int i = 0; i < n_dims; i++)
        out_fn(i + 1) = tw(i);
    out_fn(n_dims + 1) = -qw + vw_tw;
}