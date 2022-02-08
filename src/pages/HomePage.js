import React, { Component } from 'react';
import { MainLayout } from '../components/MainLayout';
import stylesHome from '../SCSS/HomePage.module.scss';
import { Button, Row, Col } from 'reactstrap';
import {say_hello, clear_it} from '../../__target__/hello.js';

export default class Home extends Component{
  constructor( props ) {
    super();

    var source=require('./HomePage.json');
    source.projects = [];

    this.state={
      source: source
    }

    this.getProjectList = this.getProjectList.bind( this );
  }

  componentDidMount() {
    document.getElementById("sayBtn").onclick = say_hello;
document.getElementById("clearBtn").onclick = clear_it;

var new_elem = document.createElement("u");
  var new_content = document.createTextNode("New Content");
  new_elem.appendChild(new_content); 
document.getElementById("root2").replaceWith(new_elem);
    window.scrollTo(0, 0);
    this.getProjectList();
  }

  componentDidUpdate( prevProps ) {
    if ( !prevProps.state.projectList.length )
      this.getProjectList();
  }

  getProjectList() {
    var source = this.state.source;
    if ( this.props.state.projectList.length > 6 ) {
      source.projectsMore= this.props.state.projectList.slice( 6, this.props.state.projectList.length );
      source.projects= this.props.state.projectList.slice( 0, 6 );
    } else 
      source.projects = this.props.state.projectList;
    this.setState({source: source})
  }

  render(){
    var source = this.state.source;
    return(
      <MainLayout isHome={ true } {...this.props } >
        <div className={ `${ stylesHome.parallax } ${ stylesHome.bannerPic }` }  id="home" >
          <div className={ stylesHome.displayMiddle } >
            <div className={ stylesHome.name } >ROBYN CHING</div> 
          </div>
        </div>
        <div id="root2">ROOT</div>
        <div id="destination"></div>
        <button type="button" id="sayBtn">Click Me!</button>
        <button type="button" id="clearBtn">Clear</button>
        <AboutSection text={ source.aboutMe.text } />
        <SYDESection text={ source.SYDE.text } />
        <PictureParallax title="projects" />
        <ProjectSection source={ source } history={ this.props.history } />
        <PictureParallax title="contact" />
        <ContactSection />
        <PictureParallax title="interests" />
        <InterestsSection interests={ source.interests } />
      </MainLayout>
    );
  }
}

class PictureParallax extends Component {
  render(){
    return(
      <div className={ `${ stylesHome.parallax } ${ stylesHome[ this.props.title.trim().toLowerCase() + "Pic" ] }` }>
        <div className={ stylesHome.displayMiddle }>
          <span className={ stylesHome.sectionName } >{ this.props.title.toUpperCase() }</span>
        </div>
      </div>
    )
  }
}

class AboutSection extends Component {
  render(){
    return(
      <div className={ stylesHome.sectionContainer} id="about">
        <h3>ABOUT ME</h3>
        <div className={ stylesHome.subTitle }>Candidate for Systems Design Engineering Class 2022</div>
        <div className={ stylesHome.floatContainer }>
          <img src={ require( '../images/robynProfile2.JPG' ) } alt="Profile" className={ stylesHome.profilePic } />
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
            <div style={{ textAlign: "left", marginBottom: "1em" }} dangerouslySetInnerHTML={ { __html: this.props.text } }/>
            <Button style={{ backgroundColor:"black", marginTop: "10px" }}>
            <a style={{marginTop: "64px", color: "white" }} href="https://github.com/rochi138/personal-website-repo/raw/master/src/documents/Robyn%20Ching%20-%20Resume.pdf" target="_blank" rel="noopener noreferrer">Haven't seen my resume yet?<br />Take a quick look!</a>
          </Button>
          </div>
        </div>
      </div>
    )
  }
}

class SYDESection extends Component {
  state={
    show: false,
  }
  render(){
    const show = this.state.show;
    return(
      <div className={ stylesHome.sectionContainer} id="whatissyde" style={{paddingTop: "0"}}>
        <h3>WHAT IS SYSTEMS DESIGN ENGINEERING?</h3>
        <div className={ stylesHome.subTitle }>Systems is how we know the world. Design is how we change it.</div>
        { show &&
          <div dangerouslySetInnerHTML={ { __html: this.props.text } }/>
        }
        <Button onClick={ () => this.setState({show: !show }) } style={{ backgroundColor: "#ccc", border: "none", color: "black" }}>
          { show ? "Show Less" : "Read More"}
        </Button>
      </div>
    )
  }
}

class ProjectSection extends Component {
  state={
    hover: undefined,
    show: false,
  }

  render(){
    const source = this.props.source;
    const show = this.state.show;
    return(
      <div className={ stylesHome.sectionContainer} id="projects">
        <h3>MY WORK</h3>
        <div className={ stylesHome.subTitle }>Previous work and personal projects.<br /> Click for the project's page</div>
        <Row>
          { source.projects.map( ( project, i ) =>
            <ProjectItem project={ project } history={ this.props.history } key={ i } i={ i } hover={ this.state.hover } setState={ ( index ) => this.setState({ hover: index }) } />
          ) }
          { ( show && source.projectsMore ) && source.projectsMore.map( ( project, i ) =>
            <ProjectItem project={ project } history={ this.props.history } key={ i } i={ i } hover={ this.state.hover } setState={ ( index ) => this.setState({ hover: index }) } />
          ) }
        </Row>
        { source.projectsMore && 
          <Button onClick={ () => this.setState({show: !this.state.show }) } style={{ backgroundColor: "#ccc", border: "none", color: "black" }}>
            { show ? "Show Less" : "Load More"}
          </Button>
        }
      </div>
    )
  }
}

class ProjectItem extends Component {
  render(){
    const project = this.props.project;
    return(
      <Col sm={ 6 } md={ 4 } onClick={ () => this.props.history.push( '/' + project.projectName ) } onMouseEnter={ () => this.props.setState( this.props.i )} onMouseLeave={ () => this.props.setState( undefined )} style={ ( this.props.hover !== undefined && this.props.i !== this.props.hover ) ? {opacity: 0.4, cursor: "pointer"} : {opacity: 1, cursor: "pointer"}}>
        <div style={{height: "16em", width: "100%", textAlign: "center"}} >
          <span style={{height: "100%", display: "inline-block", verticalAlign: "middle"}}></span>
          <img src={ require('../images/' + project.thumbnail.image + '.jpg' ) } style={{width: "90%", height: "100%", verticalAlign: "middle", objectFit: "contain"}} alt={ project.thumbnail.alt } />
        </div>
        <div style={{height: "5em", display: "flex", flexDirection: "column", alignItems: "center", marginBottom: "0.5em"}} >
          <h4>{ project.thumbnail.name }</h4>
          <p style={{textAlign: "center"}}>{ project.thumbnail.description }<br />{ project.thumbnail.date }</p>
        </div>
      </Col>
    )
  }
}

class ContactSection extends Component {
  render(){
    return(
      <div className={ stylesHome.sectionContainer} id="contact">
        <h3>Question? Comments? Concerns?</h3>
        <div className={ stylesHome.subTitle }>Let me know and I'll get back to you!</div>
        <div id="div-container">
          <div style={ { textAlign: "center" }}>
            <i className="fa fa-envelope fa-fw w3-hover-text-black w3-xlarge w3-margin-right"></i> Email: rjching@uwaterloo.ca<br />
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