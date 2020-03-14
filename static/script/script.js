$.ajax({
    success: function () {
        loadAllInputOutput(); //for embedding and extraction
        loadAccordionsAsMetric(); //to show metrics and give explanations
        $(".btn:not(.process):not(.download)").on('click', function () {
            $(this)[0].nextSibling.nextElementSibling.click();
        });

        // only show error text if there is any error
        $("#error").hide();

        //hide mode I/O when it's not selected
        $("#mode").change(function (e) {
            $(`.${e.target.options[
                e.target.selectedIndex
            ].value.toLowerCase()}`).show();
            $(`.${e.target.options[
                1 - parseInt(e.target.selectedIndex)
            ].value.toLowerCase()}`).hide();

            // disable download button
            $(`a.${e.target.options[
                e.target.selectedIndex
            ].value.toLowerCase()}.download`)[0].classList.add("disabled");

            // hide input and show primary button instead
            $("input[type='file']").hide();

            // hide accordion
            $(`.accordion.${e.target.options[
                e.target.selectedIndex
            ].value.toLowerCase()}`).hide();
        });

        // select embed at first
        $('#mode').val('Embed').change();

        // send image to server side to get preview in return
        $("input.embed.host, input.embed.wm, input.extract.wmed").change(function (e) {
            $.ajax({
                type: 'POST',
                url: `input/${e.target.classList[1]}`,
                data: e.target.files[0],
                processData: false,
                contentType: false,
                success: function (address) {
                    //display result image
                    address += ".jpg";
                    $(`img.${e.target.classList[0]}.${e.target.classList[1]}`)
                        .attr('src', Flask.url_for("static", { "filename": address }))
                        .width(300)
                        .height(300);
                }
            });
        });

        // click process will start embedding or extraction
        $("a.embed.process, a.extract.process").click(function (e) {
            // display loading
            $(this).html(
                '<div id="loading"><span class="spinner-grow spinner-grow-sm"></span>Loading..</div>'
            );
            //disable button temporarily
            $(this)[0].classList.add('disabled');

            $("#error").hide();

            $.ajax({
                type: 'POST',
                url: e.target.classList[0],
                processData: false,
                contentType: false,
                success: function (response) {
                    // enable button
                    e.target.classList.remove("disabled");

                    // remove loading
                    $('div#loading').remove();
                    e.target.text = "Process";

                    if (response.error == undefined) {
                        // show preview image
                        $(`img.${e.target.classList[0]}.${(e.target.classList[0] === 'embed') ? 'wmed' : 'wm'}`)
                            .attr(
                                'src',
                                Flask.url_for(
                                    "static",
                                    { "filename": (response.image + ".jpg") }
                                )
                            )
                            .width(300)
                            .height(300);
                        // show ssim
                        showMetrics(response);
                        $(`.accordion.${e.target.classList[0]}`).show();

                        // downloadable image
                        $(`a.${e.target.classList[0]}.download`)
                            .attr(
                                'href',
                                Flask.url_for(
                                    "static",
                                    { "filename": (response.image + ".tif") }
                                )
                            );
                        $(`a.${e.target.classList[0]}.download`)[0]
                            .classList
                            .remove("disabled");
                    } else {
                        //show error
                        $("#error").text(response.error);
                        $("#error").show();
                    }
                }
            });
        });
    }
});

// UI files functions

function getUploadFileButton(button, classname) {
    let input = $(
        "<input>",
        {
            "type": "file",
            "accept": "image/tiff",
            "class": `${button[0].classList[0]} ${classname}`,
            "style": "display: none"
        }
    );
    button.text("Upload");

    let div = $("<div>")
    div.html(`
        ${button[0].outerHTML}
        ${input[0].outerHTML}
    `);
    return div[0].outerHTML;
}

//hardcoded card form

function getCardFor(type, ioType, cardText) {
    let classname = (cardText == 'Watermark') ?
        'wm' :
        (cardText == 'Watermarked') ?
            'wmed' : 'host';

    let button1 = $("<a>", { 'class': `${type} btn btn-primary` });

    if (ioType == 'output') {
        button1.text("Process");
        button1[0].classList.add("process");
    } else {
        button1 = getUploadFileButton(button1, classname);
    }

    let button2 = (ioType == 'output') ?
        `<a class="${type} download btn btn-primary" download="image.tif">Download</a>`
        : '';

    return `
        <td>
            <div class="card ${type} ${ioType}">
                <img class="card-img-top ${type} ${classname}">
                <form>
                    <div class="${type} card-body">
                        <h4 class="${type} card-title">${cardText}</h4>
                        ${(ioType == 'output') ? button1[0].outerHTML : button1} 
                        ${button2}
                    </div>
                </form>
            </div>
        </td>
    `
}

// hardcoded accordion

function metricAccordion(type, data) {
    return `
    <div class="card accordion ${type}">
        <div class="card-header">
            <a class="card-link" data-toggle="collapse" href="#${data.id}">
            ${data.title} <span id="${data.id}Value"></span>
            </a>
        </div>
        <div id="${data.id}" class="collapse" data-parent="#accordion">
            <div class="card-body">
                <div id="${data.id}Explanation"></div>
                <p>
                    ${data.description}
                </p>
            </div>
        </div>
    </div>
    `;
}

// metrics are shown in accordion
function showMetrics(metricData) {
    $("#ssimValue").text(metricData.max);
    $("#ssimExplanation").html(
        `<p>Embedding in red: ${metricData.red}</p>
        <p>Embedding in green: ${metricData.green}</p>
        <p>Embedding in blue: ${metricData.blue}</p>`
    );
    $("#accordion").show();
}

// initialize I/O and metrics

function loadAllInputOutput() {
    loadInputOutputFor(
        'embed',
        [
            { 'IOType': 'input', 'title': 'Host Image' },
            { 'IOType': 'input', 'title': 'Watermark' },
            { 'IOType': 'output', 'title': 'Watermarked' }
        ]
    )
    loadInputOutputFor(
        'extract',
        [
            { 'IOType': 'input', 'title': 'Watermarked' },
            { 'IOType': 'output', 'title': 'Watermark' }
        ]
    );
}

function loadInputOutputFor(type, IO) {
    for (item of IO) {
        $("#files").append(getCardFor(type, item.IOType, item.title));
    }
}

function loadAccordionsAsMetric() {
    loadAccordionsFor(
        'embed',
        [
            {
                'id': 'ssim',
                'title': 'SSIM: ',
                'description': `Structural Similarity Index (SSIM) is a metric to show similarity between 
                two images with 0 means no similarity and 1 means images are perfectly similar. 
                In this project, SSIM shows similarity between original and watermarked images.`
            }
        ]
    );
}

function loadAccordionsFor(type, accordions) {
    for (item of accordions) {
        $('#accordion').append(metricAccordion(type, item));
    }
}