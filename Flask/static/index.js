// Vanguard HDI Analyzer - Interactive Frontend Engine

document.addEventListener("DOMContentLoaded", () => {
    // Country average statistics template mapping for smooth user experience
    const countryTemplates = {
        "Switzerland": { life: 84.0, meanSchool: 13.9, expSchool: 16.5, gni: 69000 },
        "Norway": { life: 86.1, meanSchool: 13.0, expSchool: 18.0, gni: 66000 },
        "Iceland": { life: 83.0, meanSchool: 12.8, expSchool: 19.0, gni: 56000 },
        "Hong Kong": { life: 85.5, meanSchool: 12.2, expSchool: 17.3, gni: 62000 },
        "Australia": { life: 83.2, meanSchool: 12.7, expSchool: 16.2, gni: 49000 },
        "United States": { life: 77.2, meanSchool: 13.7, expSchool: 16.3, gni: 64000 },
        "Japan": { life: 84.8, meanSchool: 12.9, expSchool: 15.2, gni: 42000 },
        "Germany": { life: 81.0, meanSchool: 14.1, expSchool: 17.0, gni: 54000 },
        "Brazil": { life: 76.2, meanSchool: 8.1, expSchool: 15.6, gni: 14000 },
        "India": { life: 67.2, meanSchool: 6.7, expSchool: 11.9, gni: 6600 },
        "Egypt": { life: 70.2, meanSchool: 7.4, expSchool: 13.8, gni: 12000 },
        "Kenya": { life: 61.4, meanSchool: 6.6, expSchool: 11.3, gni: 4200 },
        "Niger": { life: 62.8, meanSchool: 2.1, expSchool: 5.7, gni: 1200 }
    };

    // State Variables
    let currentSession = null;
    let countriesList = [];
    let modelsList = [];
    let activeReportPath = null;

    // DOM Elements
    const loginModal = document.getElementById("login-modal");
    const loginForm = document.getElementById("login-form");
    const appContainer = document.getElementById("app-container");
    const logoutBtn = document.getElementById("logout-btn");
    
    const displayUserName = document.getElementById("display-user-name");
    const displayUserRole = document.getElementById("display-user-role");
    const userAvatarText = document.getElementById("user-avatar-text");
    
    const selectCountry = document.getElementById("predict-country");
    const selectModel = document.getElementById("predict-model");
    
    // Sliders
    const inputLife = document.getElementById("life-expectancy");
    const inputMeanSchool = document.getElementById("mean-schooling");
    const inputExpSchool = document.getElementById("expected-schooling");
    const inputGni = document.getElementById("gnl-per-capita");
    
    const labelLife = document.getElementById("val-life");
    const labelMeanSchool = document.getElementById("val-mean-school");
    const labelExpSchool = document.getElementById("val-exp-school");
    
    // Model Info
    const modelDetailsCard = document.getElementById("model-details-card");
    const modelAlgoName = document.getElementById("model-algo-name");
    const modelAccuracy = document.getElementById("model-accuracy");
    const modelR2 = document.getElementById("model-r2");
    
    // Output Report
    const outputBlankState = document.getElementById("output-blank-state");
    const outputLoader = document.getElementById("output-loader");
    const activeReportSection = document.getElementById("active-report-section");
    const svgReportContainer = document.getElementById("svg-report-container");
    const reportPredId = document.getElementById("report-pred-id");
    const reportPredScore = document.getElementById("report-pred-score");
    const downloadReportBtn = document.getElementById("download-report-btn");
    
    // Logs Table
    const logsTableBody = document.getElementById("logs-table-body");
    const tableBlankState = document.getElementById("table-blank-state");
    const logsCount = document.getElementById("logs-count");
    
    // Prediction Form
    const predictionForm = document.getElementById("prediction-form");

    // Initialize Page
    checkSession();

    // 1. Session & Auth Functions
    function checkSession() {
        const storedSession = localStorage.getItem("hdi_session");
        if (storedSession) {
            currentSession = JSON.parse(storedSession);
            showDashboard();
        } else {
            showLogin();
        }
    }

    function showLogin() {
        loginModal.ariaHidden = false;
        loginModal.classList.remove("hidden");
        appContainer.classList.add("hidden");
    }

    function showDashboard() {
        loginModal.ariaHidden = true;
        loginModal.classList.add("hidden");
        appContainer.classList.remove("hidden");
        
        // Update user UI
        displayUserName.textContent = currentSession.name;
        displayUserRole.textContent = currentSession.role;
        userAvatarText.textContent = currentSession.name.charAt(0).toUpperCase();
        
        // Load initial metadata and past prediction logs
        loadMetaData();
        loadPredictionLogs();
    }

    // Login Submission
    loginForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const name = document.getElementById("login-name").value;
        const email = document.getElementById("login-email").value;
        const role = document.getElementById("login-role").value;
        
        const submitBtn = document.getElementById("login-submit-btn");
        submitBtn.disabled = true;
        submitBtn.innerHTML = `<span>Connecting...</span><i class="fa-solid fa-circle-notch fa-spin"></i>`;
        
        try {
            const res = await fetch("/api/auth/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name, email, role })
            });
            
            if (!res.ok) throw new Error("Authentication failed");
            
            const data = await res.json();
            localStorage.setItem("hdi_session", JSON.stringify(data));
            currentSession = data;
            showDashboard();
        } catch (err) {
            alert("Error connecting to server. Please ensure the backend is running.");
            console.error(err);
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = `<span>Initialize Session</span><i class="fa-solid fa-right-to-bracket"></i>`;
        }
    });

    // Logout
    logoutBtn.addEventListener("click", async () => {
        if (!currentSession) return;
        
        try {
            await fetch(`/api/auth/logout/${currentSession.session_id}`, { method: "POST" });
        } catch (err) {
            console.error("Error during server logout:", err);
        }
        
        localStorage.removeItem("hdi_session");
        currentSession = null;
        showLogin();
    });

    // 2. Event Listeners for Metric Sliders
    inputLife.addEventListener("input", (e) => labelLife.textContent = parseFloat(e.target.value).toFixed(1));
    inputMeanSchool.addEventListener("input", (e) => labelMeanSchool.textContent = parseFloat(e.target.value).toFixed(1));
    inputExpSchool.addEventListener("input", (e) => labelExpSchool.textContent = parseFloat(e.target.value).toFixed(1));

    // 3. Load Metadata (Countries, Models)
    async function loadMetaData() {
        try {
            // Load Countries
            const resCountries = await fetch("/api/countries");
            countriesList = await resCountries.json();
            
            selectCountry.innerHTML = '<option value="">Select country...</option>';
            countriesList.forEach(c => {
                const opt = document.createElement("option");
                opt.value = c.country_id;
                opt.textContent = `${c.country_name} (${c.region})`;
                selectCountry.appendChild(opt);
            });
            
            // Load Models
            const resModels = await fetch("/api/models");
            modelsList = await resModels.json();
            
            selectModel.innerHTML = '<option value="">Select ML engine...</option>';
            modelsList.forEach((m, idx) => {
                const opt = document.createElement("option");
                opt.value = m.model_id;
                if (idx === 0) opt.selected = true;
                opt.textContent = m.model_name;
                selectModel.appendChild(opt);
            });
            
            // Trigger model detail render for selected model
            updateModelMetadataDisplay();
            
        } catch (err) {
            console.error("Error loading metadata:", err);
        }
    }

    // Auto-fill values when Country selection changes
    selectCountry.addEventListener("change", (e) => {
        const countryId = e.target.value;
        if (!countryId) return;
        
        const selected = countriesList.find(c => c.country_id == countryId);
        if (!selected) return;
        
        const stats = countryTemplates[selected.country_name];
        if (stats) {
            inputLife.value = stats.life;
            labelLife.textContent = stats.life.toFixed(1);
            
            inputMeanSchool.value = stats.meanSchool;
            labelMeanSchool.textContent = stats.meanSchool.toFixed(1);
            
            inputExpSchool.value = stats.expSchool;
            labelExpSchool.textContent = stats.expSchool.toFixed(1);
            
            inputGni.value = stats.gni;
        }
    });

    // Model selection detail changes
    selectModel.addEventListener("change", updateModelMetadataDisplay);

    function updateModelMetadataDisplay() {
        const modelId = selectModel.value;
        if (!modelId) {
            modelDetailsCard.classList.add("hidden");
            return;
        }
        
        const selected = modelsList.find(m => m.model_id == modelId);
        if (selected) {
            modelDetailsCard.classList.remove("hidden");
            modelAlgoName.textContent = selected.algorithm_used;
            modelAccuracy.textContent = selected.accuracy_score.toFixed(3);
            modelR2.textContent = selected.r2_score.toFixed(3);
        } else {
            modelDetailsCard.classList.add("hidden");
        }
    }

    // 4. Run Predictions
    predictionForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        if (!currentSession) return;
        
        const countryId = selectCountry.value;
        const modelId = selectModel.value;
        
        if (!countryId) {
            alert("Please select a target country reference.");
            return;
        }
        if (!modelId) {
            alert("Please select a machine learning engine.");
            return;
        }
        
        const life_expectancy = parseFloat(inputLife.value);
        const mean_years_schooling = parseFloat(inputMeanSchool.value);
        const expected_years_schooling = parseFloat(inputExpSchool.value);
        const gnl_per_capita = parseFloat(inputGni.value);
        
        // Show loading state
        outputBlankState.classList.add("hidden");
        activeReportSection.classList.add("hidden");
        outputLoader.classList.remove("hidden");
        
        const submitBtn = document.getElementById("predict-submit-btn");
        submitBtn.disabled = true;
        submitBtn.innerHTML = `<span>Calculating Metrics...</span><i class="fa-solid fa-spinner fa-spin"></i>`;
        
        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    user_id: currentSession.user_id,
                    country_id: parseInt(countryId),
                    life_expectancy,
                    mean_years_schooling,
                    expected_years_schooling,
                    gnl_per_capita,
                    model_id: parseInt(modelId)
                })
            });
            
            if (!res.ok) throw new Error("Calculation API failed");
            
            const data = await res.json();
            
            // Set active report info
            reportPredId.textContent = `#${data.prediction_id}`;
            reportPredScore.textContent = data.score.toFixed(4);
            activeReportPath = data.graph_path;
            
            // Load and inline the SVG XML for premium interactivity & display
            const resSvg = await fetch(data.graph_path);
            const svgText = await resSvg.text();
            svgReportContainer.innerHTML = svgText;
            
            // Reveal report
            outputLoader.classList.add("hidden");
            activeReportSection.classList.remove("hidden");
            
            // Reload logs table
            loadPredictionLogs();
            
        } catch (err) {
            alert("Prediction failed. Please check connection.");
            console.error(err);
            outputLoader.classList.add("hidden");
            outputBlankState.classList.remove("hidden");
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = `<span>Compute HDI & Report</span><i class="fa-solid fa-microchip"></i>`;
        }
    });

    // 5. Load Prediction Logs
    async function loadPredictionLogs() {
        try {
            const res = await fetch("/api/predictions");
            const predictions = await res.json();
            
            logsCount.textContent = `${predictions.length} ${predictions.length === 1 ? 'Record' : 'Records'}`;
            
            if (predictions.length === 0) {
                logsTableBody.innerHTML = "";
                tableBlankState.classList.remove("hidden");
                return;
            }
            
            tableBlankState.classList.add("hidden");
            logsTableBody.innerHTML = "";
            
            predictions.forEach(p => {
                const tr = document.createElement("tr");
                
                // Get classification badge style
                let catClass = "cat-badge-low";
                if (p.hdi_category.includes("Very High")) catClass = "cat-badge-veryhigh";
                else if (p.hdi_category.includes("High")) catClass = "cat-badge-high";
                else if (p.hdi_category.includes("Medium")) catClass = "cat-badge-medium";
                
                tr.innerHTML = `
                    <td>${p.prediction_time}</td>
                    <td>
                        <div style="font-weight: 600;">${p.user.name}</div>
                        <div style="font-size: 11px; color: var(--text-muted);">${p.user.role}</div>
                    </td>
                    <td>
                        <div style="font-weight: 600;">${p.country.name}</div>
                        <div style="font-size: 11px; color: var(--text-muted);">${p.country.region}</div>
                    </td>
                    <td>
                        <div style="font-weight: 600;">${p.model.algorithm}</div>
                        <div style="font-size: 11px; color: var(--text-muted);">${p.model.name}</div>
                    </td>
                    <td class="col-metrics">
                        <i class="fa-solid fa-heart-pulse text-accent" title="Life Expectancy"></i> Live-Ex | 
                        <i class="fa-solid fa-graduation-cap text-accent" title="Education Index"></i> Edu-Idx
                    </td>
                    <td class="log-score text-accent">${p.predicted_hdi_score.toFixed(4)}</td>
                    <td>
                        <span class="cat-badge ${catClass}">${p.hdi_category}</span>
                    </td>
                    <td>
                        <button class="btn btn-secondary btn-sm load-log-report-btn" data-path="${p.report_path}" data-id="${p.prediction_id}" data-score="${p.predicted_hdi_score}">
                            <i class="fa-solid fa-chart-line"></i> View
                        </button>
                    </td>
                `;
                
                // Wire up click event for loading specific prediction
                tr.querySelector(".load-log-report-btn").addEventListener("click", async (e) => {
                    const btn = e.currentTarget;
                    const path = btn.getAttribute("data-path");
                    const id = btn.getAttribute("data-id");
                    const score = parseFloat(btn.getAttribute("data-score"));
                    
                    // Show loader briefly to indicate click action
                    outputBlankState.classList.add("hidden");
                    activeReportSection.classList.add("hidden");
                    outputLoader.classList.remove("hidden");
                    
                    try {
                        const resSvg = await fetch(path);
                        const svgText = await resSvg.text();
                        svgReportContainer.innerHTML = svgText;
                        
                        reportPredId.textContent = `#${id}`;
                        reportPredScore.textContent = score.toFixed(4);
                        activeReportPath = path;
                        
                        outputLoader.classList.add("hidden");
                        activeReportSection.classList.remove("hidden");
                        
                        // Scroll up smoothly to view the report card
                        document.querySelector(".card-output-panel").scrollIntoView({ behavior: "smooth" });
                    } catch (err) {
                        alert("Error loading historical SVG report.");
                        console.error(err);
                        outputLoader.classList.add("hidden");
                        outputBlankState.classList.remove("hidden");
                    }
                });
                
                logsTableBody.appendChild(tr);
            });
            
        } catch (err) {
            console.error("Error loading predictions logs:", err);
        }
    }

    // 6. Download SVG report
    downloadReportBtn.addEventListener("click", () => {
        if (!activeReportPath) return;
        
        const link = document.createElement("a");
        link.href = activeReportPath;
        link.download = activeReportPath.split("/").pop();
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
});
