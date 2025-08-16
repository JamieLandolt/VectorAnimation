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
                if (jobIsDone) {
                    clearInterval(intervalId);
                    intervalId = null;
                }
            })
    }

    function sendJobCreationRequest() {
        if (points.length == 0) return;

        jobIsDone = false;

        const payload = {
            "points": points
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
                setInterval(sendJobInfoRequest, 30000);
            })
    }

    generateBtn.addEventListener("click", sendJobCreationRequest);
})

