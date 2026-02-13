$(document).ready(function(){
    let div_logo = document.getElementsByClassName("icon")[0];
    console.log(div_logo);
    console.log(div_logo.text);
    div_logo.href = "https://github.com/yudi-mars/Soul";
    div_logo.target = "_blank";
    div_logo.text = "    Soul    ";
});
