import React, { Component } from 'react';
import styles from '../SCSS/Main.module.scss';

export default class Footer extends Component{
  render(){
    return(
    	<footer className={ styles.footer }>
        Robyn Ching
        <div style={{ display: "flex" }}>
          <a href="https://github.com/rochi138" target="_blank" rel="noopener noreferrer"><i className="fab fa-github" /></a><br />
          <a href="https://www.linkedin.com/in/robyn-ching/" target="_blank" rel="noopener noreferrer"><i className="fab fa-linkedin" /></a>
        </div>
      </footer>
    	)
	}
}