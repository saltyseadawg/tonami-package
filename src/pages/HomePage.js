import React, { Component } from 'react';
import {say_hello, clear_it} from '../../__target__/hello.js';

export default class Home extends Component{
  constructor( props ) {
    super( props );
  }

  componentDidMount() {
    document.getElementById("sayBtn").onclick = say_hello;
    document.getElementById("clearBtn").onclick = clear_it;

    var new_elem = document.createElement("u");
    var new_content = document.createTextNode("New Content");
    new_elem.appendChild(new_content); 
    document.getElementById("root2").replaceWith(new_elem);
  }

  render(){
    return(
      <div>
        <div>space space</div>
        <div>space space</div>
        <div id="root2">ROOT</div>
        <div id="destination"></div>
        <button id="sayBtn">Click Me!</button>
        <button id="clearBtn">Clear</button>
        <button id="to1" onClick={() => this.props.changePage(1)}>1</button>
        <button id="to2" onClick={() => this.props.changePage(2)}>2</button>
      </div>
    );
  }
}
