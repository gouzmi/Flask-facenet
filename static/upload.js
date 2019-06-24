$("#file-picker").change(function(){
    var input = document.getElementById('file-picker');
    for (var i=0; i<input.files.length; i++)
    {
        var ext= input.files[i].name.substring(input.files[i].name.lastIndexOf('.')+1).toLowerCase()
        if ((ext == 'jpg') || (ext == 'png') || (ext=='jpeg'))
        {
            $("#msg").text("Files supported")
        }
        else
        {
            $("#msg").text("Files NOT supported")
            document.getElementById("file-picker").value ="";
        }
    }
} );

function readURL(input) {
if (input.files && input.files[0]) {

var reader = new FileReader();

reader.onload = function(e) {
  $('.image-upload-wrap').hide();

  $('.file-upload-image').attr('src', e.target.result);
  $('.file-upload-content').show();

  $('.image-title').html(input.files[0].name);
};

reader.readAsDataURL(input.files[0]);

} else {
removeUpload();
}
}

function removeUpload() {
$('.file-upload-input').replaceWith($('.file-upload-input').clone());
$('.file-upload-content').hide();
$('.image-upload-wrap').show();
}
$('.image-upload-wrap').bind('dragover', function () {
    $('.image-upload-wrap').addClass('image-dropping');
});
$('.image-upload-wrap').bind('dragleave', function () {
    $('.image-upload-wrap').removeClass('image-dropping');
});
