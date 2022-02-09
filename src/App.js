import React, { Component } from 'react';
import { Route } from 'react-router-dom';
import './SCSS/App.scss';
import Home from './pages/HomePage';

export default class App extends Component {
  constructor(props) {
		super();

		this.state={
      theme: "default"
		}
  }

  render() {
    return (
      <div>
        <Route exact path='/' render={(props) => <Home state={this.state} {...props} setState={ (state) => this.setState( state ) }/>} />
        <Route exact path='/1' render={(props) => <Home state={this.state} {...props} setState={ (state) => this.setState( state ) }/>} />
        <Route exact path='/2' render={(props) => <Home state={this.state} {...props} setState={ (state) => this.setState( state ) }/>} />
        <Route exact path='/3' render={(props) => <Home state={this.state} {...props} setState={ (state) => this.setState( state ) }/>} />
      </div>
    );
  }
}
