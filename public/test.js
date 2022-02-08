import {say_hello, clear_it} from './__target__/hello.js';

document.getElementById("sayBtn").onclick = say_hello;
document.getElementById("clearBtn").onclick = clear_it;

var new_elem = document.createElement("u");
  var new_content = document.createTextNode("New Content");
  new_elem.appendChild(new_content); 
document.getElementById("root2").replaceWith(new_elem);