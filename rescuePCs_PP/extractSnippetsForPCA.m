% function cProjPC = extractSnippetsForPCA(rez, nPCs, nChannels)
% % extracts principal components for 1D snippets of spikes from all channels
% % loads a subset of batches to find these snippets
% 
% % PP: this works, but does not do what I need.
% % I need to use the actual spike timestamps, within each batch (skipping is
% % fine), instead of finding isolate peaks anew (I need to keep a reference
% % to the given cluster. I guess I will have to do this for every cluster,
% % though doing it in parallel would be a lot faster. As long as I don't
% % loose clust ID ????
% 
% % isolated_peaks_new finds all the peaks, and for each of them gives the
% % row (timepoint in batch), column (best channel), mu (probably amplitude?)
% 
% % get_SpikeSample gets the wf clip centered (check) around the points
% % specified in row, col. Perhaps it can extract multiple channels at once
% % (dimension of output here is 61,1,nspikes) - check better
% 
% %check output again:
% % pc_features = readNPY('D:\paola\sampleForPhy\M1_102721_SpikeData\rez2phy_resave\pc_features.npy');
% % spclu = readNPY('D:\paola\Dropbox\dati\Apr2021\Rita\A333_09032021\spike_clusters.npy');
% % pc_feature_ind = readNPY('D:\paola\sampleForPhy\M1_102721_SpikeData\rez2phy_resave\pc_feature_ind.npy');
debug = 1
if debug
    disp('you are in debug mode')
    cd('/Users/galileo/Dropbox/Data/M1_102721_SpikeData') 
    edit clusterSingleBatches.m %this might be useful, in relationship to the call of the next one:
    edit extractPCbatch2  %this might be useful
    edit learnAndSolve8b_old %check
end
st = readNPY('spike_times.npy');
clu = readNPY('spike_clusters.npy');  %make sure you in the right folder...
load('rez.mat')
% I will simply calculate the best channels after loading the spikes, for
% each one

rez.ops.fproc = fullfile(pwd, 'temp_wh.dat');
ops = rez.ops;
Nbatch      = rez.temp.Nbatch;
NT  	= ops.NT; %timpoints within each batch
batchstart = 0:NT:NT*Nbatch;

nPCs = 3;
NchanNear = min(ops.Nchan, 2*4+1);
wPCA    = extractPCfromSnippets(rez, nPCs); % extract PCA waveforms pooled over channels

% extract the PCA projections
CC = zeros(ops.nt0); % initialize the covariance of single-channel spike waveforms
fid = fopen('temp_wh.dat', 'r'); % open the preprocessed data file

for ibatch = 1:100:Nbatch % from every 100th batch
    offset = 2 * ops.Nchan*batchstart(ibatch);
    fseek(fid, offset, 'bof');
    dat = fread(fid, [NT ops.Nchan], '*int16');

    % move data to GPU and scale it back to unit variance
    dataRAW = gpuArray(dat);
    dataRAW = single(dataRAW);
    dataRAW = dataRAW / ops.scaleproc;
    
    % report st to given batch
    stBatch = st(st > batchstart(ibatch) & st <= batchstart(ibatch+1));
    % I need to know at least the best channel for the given template, or I
    % might confuse among clusters. and then I will limit to the nearest 8
    % channels (seems sufficient). I could limit furhter. Sort channels
    % based on distance only. So I can prepare this info beforehand for all
    % template. Do I have this info over time?
    channels2use; % limiting step, but doable!! make it



    % find isolated spikes from each batch
%     [row, col, mu] = get_st_in_batch_gpu(dataRAW, stBatch, channels2use);% I DO NOT NEED THIS - I ALREADY HAVE THE COORDINATES
%     row is a column vector of all points in time (in batch)
%     col is channel to extract
% mu I don't need
    row = repmat(stBatch', nChannels, 1);
    row = row(:); %all set

    % for each peak, get the voltage snippet from that channel
    clips = get_SpikeSample(dataRAW, row, col, ops, 0);

    c = sq(clips(:, :));
    CC = CC + gather(c * c')/1e3; % scale covariance down by 1,000 to maintain a good dynamic range
end
fclose(fid);

[U Sv V] = svdecon(CC); % the singular vectors of the covariance matrix are the PCs of the waveforms

wPCA = U(:, 1:nPCs); % take as many as needed

wPCA(:,1) = - wPCA(:,1) * sign(wPCA(21,1)); % adjust the arbitrary sign of the first PC so its negativity is downward

