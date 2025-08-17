window.addEventListener("load", () => {
    // CANVAS /////////////////////////////////////////////////////////////////////////////////////
    const canvas = document.querySelector("#canvas");
    const context = canvas.getContext("2d");

    let painting = false;
    let points = [];

    function clearCanvas() {
        context.clearRect(0, 0, canvas.width, canvas.height);
        points = [];
    }

    function startPosition(e) {
        painting = true;
        context.beginPath();
        clearCanvas();
        const rect = canvas.getBoundingClientRect();
        context.moveTo(e.clientX - rect.left, e.clientY - rect.top);
    }

    function finishPosition() {
        painting = false;
        context.beginPath();
    }

    function draw(e) {
        if (!painting) return;

        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        context.lineWidth = 2;
        context.lineCap = "round";
        context.lineTo(x, y);
        context.stroke();
        context.beginPath();
        context.moveTo(x, y);

        // Add to points array
        points.push({
            x: x,
            y: y,
            timestamp: Date.now()
        });
    }

    canvas.addEventListener("mousedown", startPosition);
    canvas.addEventListener("mouseup", finishPosition);
    canvas.addEventListener("mousemove", draw);

    // CLEAR BUTTON //////////////////////////////////////////////////////////////////////////////
    const clearBtn = document.querySelector("#clear-btn");
    clearBtn.addEventListener("click", clearCanvas)

    // GENERATE BUTTON ///////////////////////////////////////////////////////////////////////////
    let jobId = null;
    let jobIsDone = false;
    let intervalId = null;
    const endpoint = "http://127.0.0.1:5000/api/job";

    const generateBtn = document.querySelector("#generate-btn");
    const canvasSection = document.querySelector("#canvas-section");
    const loadingSection = document.querySelector("#loading-section");
    const videoSection = document.querySelector("#video-section");
    const videoSectionContainer = document.querySelector("#video-section .container");

    const videoDirectory = "/media/videos/animate_from_file/1440p60/";

    function sendJobInfoRequest() {
        if (jobId == null) return;

        const params = {
            "job_id": jobId
        }

        const queryString = new URLSearchParams(params).toString();
        const endpointWithParams = `${endpoint}?${queryString}`;

        fetch(endpointWithParams, {
            method: 'GET',
        })
            .then(response => {
                if (!response.ok) console.log("Bad request");
                else {
                    return response.json();
                }
            })
            .then(data => {
                jobIsDone = data["is_done"];

                if (jobIsDone == true) {
                    console.log(intervalId);
                    clearInterval(intervalId);
                    intervalId = null;
                    loadingSection.classList.toggle("hide");
                    videoSection.classList.toggle("hide");

                    jobId = data["job_id"];

                    const videoElement = document.createElement('video');
                    const videoEndpoint = `http://127.0.0.1:5000/api/job/video?${queryString}`;
                    videoElement.src = videoEndpoint;
                    videoElement.controls = true;
                    videoSectionContainer.appendChild(videoElement);
                }
            })
    }

    function sendJobCreationRequest() {
        if (points.length == 0) return;

        jobIsDone = false;
        const numVectors = parseInt(document.querySelector("#numVectors").value, 10);
        const sortStyle = document.querySelector("#sortStyle").value;
        const sortAscending = document.querySelector("#sortAscending").checked; // .checked returns a boolean
        const keepTail = document.querySelector("#keepTail").checked;

        const n = Math.floor(numVectors / 2);
        let vectorJMin, vectorJMax;

        if (numVectors % 2 === 1) {
            // If odd (e.g., 41), create a symmetric range like -20 to 20
            vectorJMin = -n;
            vectorJMax = n;
        } else {
            // If even (e.g., 40), create a nearly symmetric range like -20 to 19
            vectorJMin = -n;
            vectorJMax = n - 1;
        }

        const payload = {
            "points": points,
            "render_params": {
                "vectorJMin": vectorJMin,
                "vectorJMax": vectorJMax,
                "sortStyle": sortStyle,
                "sortAscending": sortAscending,
                "keepTail": keepTail
            }
        }

        fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        })
            .then(response => {
                if (!response.ok) console.log("Bad request");
                else {
                    return response.json();
                }
            })
            .then(data => {
                jobId = data["job_id"];
                intervalId = setInterval(sendJobInfoRequest, 30000);

                // hide canvas section, show loading section
                canvasSection.classList.toggle("hide");
                loadingSection.classList.toggle("hide");
            })
    }

    generateBtn.addEventListener("click", sendJobCreationRequest);
})

