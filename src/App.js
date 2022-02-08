import React, { Component } from 'react';
import { Route } from 'react-router-dom';
import './SCSS/App.scss';
import Home from './pages/HomePage';
import ProjectPage from './pages/ProjectPage';
import { ProjectList } from './components/Constants';

export default class App extends Component {
  constructor(props) {
		super();

		this.state={
      theme: "default",
      projectList: []
		}
  }

  componentDidMount() {
    var rawProjectList = [];
    ProjectList.forEach( function( i ) {
      var pageMeta = require('./components/Project JSONs/' + i + '.json');
      var date = "";
      if ( pageMeta.thumbnail.date )
        date = pageMeta.thumbnail.date;
      else
        pageMeta.components.forEach( function( i ) {
          if ( i.title === "Progress" )
            date = i.list[0].left;
        })
      rawProjectList.push({ projectName: i, thumbnail: { ...pageMeta.thumbnail, date: date } })
    })
    var projectList=[ rawProjectList[0] ];
    const rlength = rawProjectList.length;
    for ( var i = 1; i < rlength; ++i ){
      const length = projectList.length;
      var added = false;
      for ( var ii = 0; ii < length; ++ii ){
        if ( rawProjectList[ i ].thumbnail.date > projectList[ ii ].thumbnail.date ){
          projectList.splice(ii, 0, rawProjectList[ i ] )
          added = true;
          break;
        }
      }
      if ( !added ) projectList.push(rawProjectList[ i ])
    }
    this.setState({ projectList: projectList });
  }

  render() {
    return (
      <div>
        <Route exact path='/' render={(props) => <Home state={this.state} {...props} setState={ (state) => this.setState( state ) }/>} />
        { this.state.projectList.map((project, i) => 
          <Route path={ '/'+ project.projectName } key={i} render={(props) => <ProjectPage fileName={project.projectName} state={this.state} i={ i } {...props} setState={ (state) => this.setState( state ) } />} />
        )}
      </div>
    );
  }
}
