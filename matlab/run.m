function f = run(file_name, flac_name, dest_dir)

pkg load signal;
[grp_phase, cep, ts] = modified_group_delay_feature(file_name, 0.1, 0.3);
% f = [grp_phase, cep, ts];
save(strcat(dest_dir, "/", flac_name, ".mat"), "grp_phase");
% save("cep.mat", "cep");
% save("ts.mat", "ts");