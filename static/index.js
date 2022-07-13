const btnSubmit = document.getElementById("btn-testing")

btnSubmit.addEventListener("click", function (e) {
    e.preventDefault()
    testing()
})

function testing(){
    var formData = new FormData()
    formData.append("berita", document.getElementById("berita").value)


    var xhr = new XMLHttpRequest();
    xhr.open('POST', 'http://127.0.0.1:5000/classify', true);
    xhr.onload = function () {
        data = JSON.parse(this.responseText)
        document.getElementById("jenisKlasifikasi").value = data.data
    };
    xhr.send(formData);

    
}