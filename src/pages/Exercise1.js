import React, { Component } from 'react';
import stylesHome from '../SCSS/HomePage.module.scss';
import { Button, Row, Col } from 'reactstrap';

export default class Exercise1 extends Component{
  constructor() {
    super();

    var source=require('../JSONs/HomePage.json');

    this.state={
      source: source
    }

  }

  render(){
    var source = this.state.source;
    return(
      <AboutSection text={ source.aboutMe.text } />
    );
  }
}

class AboutSection extends Component {
  render(){
    return(
      <div className={ stylesHome.sectionContainer} id="about">
        <h3>ABOUT ME</h3>
        <div className={ stylesHome.subTitle }>Candidate for Systems Design Engineering Class 2022</div>
        <div className={ stylesHome.floatContainer }>
          <img src={ require( '../images/placeholder.jpg' ) } alt="Profile" className={ stylesHome.profilePic } />
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
            <div style={{ textAlign: "left", marginBottom: "1em" }} dangerouslySetInnerHTML={ { __html: this.props.text } }/>
            <Button style={{ backgroundColor:"black", marginTop: "10px" }}>
            <a style={{marginTop: "64px", color: "white" }}>Haven't seen my resume yet?<br />Take a quick look!</a>
          </Button>
          </div>
        </div>
      </div>
    )
  }
}
