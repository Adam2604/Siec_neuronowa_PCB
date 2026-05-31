% =======================================================
% PROJEKT: GENERATOR DANYCH DLA SIECI NEURONOWEJ - GRUPA 15
% =======================================================

% --- KROK 1: GENEROWANIE SIATKI WSZYSTKICH 3200 KOMBINACJI ---
val_distance = linspace(200, 2000, 20); % Zmienna 1: 20 próbek
val_separation = linspace(200, 400, 4);   % Zmienna 2: 4 próbki
val_shift = linspace(0, 10000, 40);       % Zmienna 4: 40 próbek

wszystkie_kombinacje = zeros(3200, 3);
idx = 1;
for d = 1:length(val_distance)
    for s = 1:length(val_separation)
        for sh = 1:length(val_shift)
            wszystkie_kombinacje(idx, 1) = val_distance(d);
            wszystkie_kombinacje(idx, 2) = val_separation(s);
            wszystkie_kombinacje(idx, 3) = val_shift(sh);
            idx = idx + 1;
        end
    end
end

% --- KROK 2: USTAWIENIA PACZKI SYMULACJI ---
% UWAGA: Zmieniaj te dwie wartości przed każdym uruchomieniem!
% Na próbę ustawione jest przeliczenie `pierwszych 5 symulacji.
start_idx = 2301;      
end_idx = 2700;      

% Stała wymuszona dla Grupy 15
diff_line_length = 19500; 

liczba_symulacji_dzisiaj = end_idx - start_idx + 1;
dane_wynikowe = zeros(liczba_symulacji_dzisiaj, 5);
wynik_idx = 1;

fprintf('Rozpoczynam obliczenia paczki od indeksu %d do %d (łącznie %d symulacji)...\n', start_idx, end_idx, liczba_symulacji_dzisiaj);

% --- KROK 3: GŁÓWNA PĘTLA OBLICZENIOWA ---
for i = start_idx:end_idx
    
    % Pobranie wymiarów dla bieżącej symulacji
    diff_line_distance = wszystkie_kombinacje(i, 1);
    separation = wszystkie_kombinacje(i, 2);
    diff_line_shift_from_edge = wszystkie_kombinacje(i, 3);
    
    fprintf('Postęp: %d/3200 | Dist: %.0f, Sep: %.0f, Shift: %.0f\n', i, diff_line_distance, separation, diff_line_shift_from_edge);
    
    % Czyszczenie zmiennych z poprzedniego kroku (zapobiega błędom pamięci)
    clear p port mesh CSX FDTD f s11 s21 s31 s431 status message messageid;
    close all;

    % =======================================================
    % KOD OPENEMS (Zmodyfikowany do działania w pętli)
    % =======================================================
    physical_constants;
    unit = 1e-6; % specify everything in um

    substrate_length     = 30000;
    substrate_width      = 12000;
    air_spacer           = 4000;  % air spacer above the substrate

    msl_width               = 500;
    msl_substrate_thickness = 254;

    strip_width               = 500;
    strip_substrate_thickness = 512;

    connect_via_rad =  500/2;
    connect_via_gap = 1250/2;

    substrate_epr    = 3.66;
    substrate_kappa  = 1e-3 * 2*pi*2.45e9 * EPS0*substrate_epr; % substrate losses

    f_max = 5e9;

    resolution = 3*10^8/f_max/substrate_epr/40*10^6;
    edge_res   = resolution/10;
    feed_shift = 10*resolution;
    meas_shift = diff_line_length*2/3;

    % Setup FDTD Parameters & Excitation Function
    FDTD = InitFDTD('EndCriteria',10^-4,'NrTS', 0.2*10^5);
    FDTD = SetGaussExcite( FDTD, f_max/2, f_max/2);
    BC   = {'PML_8' 'PML_8' 'PML_8' 'PML_8' 'PEC' 'PML_8'};
    FDTD = SetBoundaryCond( FDTD, BC );

    % Setup CSXCAD Geometry & Mesh
    CSX = InitCSX();
    edge_mesh  = [-1/3 2/3]*edge_res; % 1/3 - 2/3 rule for 2D metal edges

    mesh.x = SmoothMeshLines( [-connect_via_gap 0 connect_via_gap], 2*edge_res, 1.5 );
    mesh.x = SmoothMeshLines( [-substrate_length/2 mesh.x substrate_length/2], resolution, 1.5);
    mesh.y = SmoothMeshLines( [0 msl_width/2+edge_mesh substrate_width/2], resolution/4 , 1.5);
    mesh.y = sort(unique([-mesh.y mesh.y]));
    mesh.z = SmoothMeshLines( [linspace(-strip_substrate_thickness,0,5) linspace(0,strip_substrate_thickness,5) linspace(strip_substrate_thickness,msl_substrate_thickness+strip_substrate_thickness,5) 2*strip_substrate_thickness+air_spacer] , resolution );
    CSX = DefineRectGrid( CSX, unit, mesh );

    % Create Substrate
    CSX = AddMaterial( CSX, 'RO4350B' );
    CSX = SetMaterialProperty( CSX, 'RO4350B', 'Epsilon', substrate_epr, 'Kappa', substrate_kappa );
    start_pt = [mesh.x(1),   mesh.y(1),   -strip_substrate_thickness];
    stop_pt  = [mesh.x(end), mesh.y(end), +strip_substrate_thickness+msl_substrate_thickness];
    CSX = AddBox( CSX, 'RO4350B', 0, start_pt, stop_pt );

    % Create a PEC called 'metal' and 'gnd'
    CSX = AddMetal( CSX, 'gnd' );
    CSX = AddMetal( CSX, 'signal' );

    % Create input EMI strip line port
    start_pt = [-substrate_length/2 -strip_width/2  0];
    stop_pt  = [0            +strip_width/2  0];
    [CSX,port{1}] = AddStripLinePort( CSX, 100, 1, 'signal', start_pt, stop_pt, strip_substrate_thickness, 'x', [0 0 -1], 'ExcitePort', true, 'FeedShift', feed_shift, 'MeasPlaneShift', meas_shift );

    % Create MSL port on top
    start_pt = [substrate_length/2  -strip_width/2 strip_substrate_thickness+msl_substrate_thickness];
    stop_pt  = [0            +strip_width/2 strip_substrate_thickness];
    [CSX,port{2}] = AddMSLPort( CSX, 100, 2, 'signal', start_pt, stop_pt, 'x', [0 0 -1], 'MeasPlaneShift', meas_shift );

    % Create 1st port from differential pair on top
    start_pt = [substrate_length/2-diff_line_shift_from_edge  -3*msl_width/2-diff_line_distance strip_substrate_thickness+msl_substrate_thickness];
    stop_pt  = [substrate_length/2-diff_line_shift_from_edge-diff_line_length            -msl_width/2-diff_line_distance strip_substrate_thickness];
    [CSX,port{3}] = AddMSLPort( CSX, 100, 3, 'signal', start_pt, stop_pt, 'x', [0 0 -1], 'MeasPlaneShift', meas_shift );
    
    % Create 2nd port from differential pair on top
    start_pt = [substrate_length/2-diff_line_shift_from_edge  -5*msl_width/2-diff_line_distance-separation strip_substrate_thickness+msl_substrate_thickness];
    stop_pt  = [substrate_length/2-diff_line_shift_from_edge-diff_line_length            -3*msl_width/2-diff_line_distance-separation strip_substrate_thickness];
    [CSX,port{4}] = AddMSLPort( CSX, 100, 4, 'signal', start_pt, stop_pt, 'x', [0 0 -1], 'MeasPlaneShift', meas_shift );

    % transitional via
    start_pt = [0, 0, 0];
    stop_pt  = [0, 0, strip_substrate_thickness+msl_substrate_thickness];
    CSX = AddCylinder(CSX, 'signal', 100, start_pt, stop_pt, connect_via_rad);

    % metal plane between strip line and MSL
    p(1,1) = mesh.x(1);
    p(2,1) = mesh.y(1);
    p(1,2) = 0;
    p(2,2) = mesh.y(1);
    a = linspace(-pi, pi, 61);
    for j=1:length(a)
        p(1,end+1) = connect_via_gap*sin(a(j));
        p(2,end)   = connect_via_gap*cos(a(j));
    end
    p(1,end+1) = 0;
    p(2,end  ) = mesh.y(1);
    p(1,end+1) = mesh.x(end);
    p(2,end  ) = mesh.y(1);
    p(1,end+1) = mesh.x(end);
    p(2,end  ) = mesh.y(end);
    p(1,end+1) = mesh.x(1);
    p(2,end  ) = mesh.y(end);
    CSX = AddPolygon( CSX, 'gnd', 1, 'z', strip_substrate_thickness, p);

    % Write/Run openEMS
    Sim_Path = 'tmp';
    Sim_CSX = 'strip2msl.xml';

    % Wyciszone komunikaty o usuwaniu folderów
    [~, ~, ~] = rmdir( Sim_Path, 's' ); 
    [~, ~, ~] = mkdir( Sim_Path ); 

    WriteOpenEMS( [Sim_Path '/' Sim_CSX], FDTD, CSX );
    
    % ZAKOMENTOWANO RYSOWANIE MODELU - żeby nie otwierać 3200 okien!
    % CSXGeomPlot( [Sim_Path '/' Sim_CSX] ); 
    
    % Uruchomienie symulatora (zablokowano drukowanie pełnego logu na ekran dla czytelności)
    RunOpenEMS( Sim_Path, Sim_CSX, '' );

    % Post-Processing
    f = linspace( 0, f_max, 1601 );
    port = calcPort( port, Sim_Path, f, 'RefImpedance', 50);

    s31 = port{3}.uf.ref./ port{1}.uf.inc;
    s431 = (port{3}.uf.ref-port{4}.uf.ref)./port{1}.uf.inc;

    % ZAKOMENTOWANO RYSOWANIE WYKRESÓW - liczy się tylko wyciągnięcie danych liczbowych!
    % plot(...);

    s31_dB_av = 1/length(s31)*20*log10(abs(s31))*ones(length(s31),1);
    s431_dB_av = 1/length(s431)*20*log10(abs(s431))*ones(length(s431),1);
    diff_processing_gain = s31_dB_av - s431_dB_av;

    % =======================================================
    % KONIEC KODU OPENEMS
    % =======================================================
    
    % Zapis do naszej matrycy dzisiejszych wyników (tylko pierwsza uśredniona wartość wektora)
    dane_wynikowe(wynik_idx, 1) = diff_line_distance;
    dane_wynikowe(wynik_idx, 2) = separation;
    dane_wynikowe(wynik_idx, 3) = diff_line_shift_from_edge;
    dane_wynikowe(wynik_idx, 4) = s431_dB_av(1);
    dane_wynikowe(wynik_idx, 5) = diff_processing_gain(1);
    
    wynik_idx = wynik_idx + 1;
end

% --- KROK 4: ZAPIS PACZKI DO PLIKU CSV ---
nazwa_pliku = sprintf('dane_grupa15_czesciowe_%d_do_%d.csv', start_idx, end_idx);
csvwrite(nazwa_pliku, dane_wynikowe);

fprintf('\n=== ZAKOŃCZONO SUKCESEM! ===\n');
fprintf('Plik z wynikami zapisany jako: %s\n', nazwa_pliku);