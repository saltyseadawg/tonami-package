import React, { Component } from 'react';
import { MainLayout } from '../components/MainLayout';
import stylesHome from '../SCSS/HomePage.module.scss';
import { Button, Row, Col } from 'reactstrap';
import {say_hello, clear_it} from '../../__target__/hello.js';

export default class Home extends Component{
  constructor( props ) {
    super();

    var source=require('./HomePage.json');

    this.state={
      source: source
    }

  }

  componentDidMount() {
    document.getElementById("sayBtn").onclick = say_hello;
    document.getElementById("clearBtn").onclick = clear_it;

    var new_elem = document.createElement("u");
    var new_content = document.createTextNode("New Content");
    new_elem.appendChild(new_content); 
    document.getElementById("root2").replaceWith(new_elem);

    window.scrollTo(0, 0);
  }

  render(){
    var source = this.state.source;
    return(
      <MainLayout isHome={ true } {...this.props } >
        <div>space space</div>
        <div>space space</div>
        <div id="root2">ROOT</div>
        <div id="destination"></div>
        <button type="button" id="sayBtn">Click Me!</button>
        <button type="button" id="clearBtn">Clear</button>
        <AboutSection text={ source.aboutMe.text } />
        <InterestsSection interests={ source.interests } />
      </MainLayout>
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

class InterestsSection extends Component {
  state={
    hover: undefined,
    selectedInterest: 0
  }
  render(){
    var openInterest = this.props.interests[ this.state.selectedInterest ];
    return(
      <div className={ stylesHome.sectionContainer} id="interests">
        <h3>OUTSIDE OF OFFICE HOURS</h3>
        <div className={ stylesHome.subTitle }>Conversation Starters <br />Things Nobody Asked For<br />Weak Flexes</div>
        <div className={ stylesHome.interestsWrapper }>
          <div className={ stylesHome.interestsBar }>
            { this.props.interests.map( ( interest, i ) =>
              <div key={ i } className={ stylesHome.option } onClick={ () => this.setState({ selectedInterest: i }) } onMouseEnter={ () => this.setState({ hover: i })} onMouseLeave={ () => this.setState({ hover: undefined })} style={ ( this.state.hover !== undefined && i!== this.state.hover ) ? {opacity: 0.4} : {opacity: 1}}>
                <img src={ require('../images/' + interest.image + '.jpg' ) } style={{width: "20%", height: "100%", objectFit: "contain", minWidth: "4em", padding: "5px"}} alt={ interest.alt } />
                <div className={ `${ stylesHome.text } ${ stylesHome.hideMobile }` }>
                  <h4>{ interest.brief }</h4>
                </div>
              </div>
            ) }
          </div>
          <div className={ stylesHome.openInterests }>
            <img src={ require('../images/' + openInterest.image + '.jpg' ) } alt={ openInterest.alt } style={{ width: "200px", height: "auto", objectFit: "contain", padding: "1em" }} />
            <div className={ stylesHome.text }>
              <h4>{ openInterest.brief }</h4>
              <div dangerouslySetInnerHTML={ { __html: openInterest.content } }/>
            </div>
          </div>
        </div>
      </div>
    )
  }
}